# model_marketplace.config
# {
#     "token_length": "128000",
#     "sampling_frequency": "44100",
#     "framework": "transformers",
#     "dataset_format": "llm",
#     "dataset_sample": "[id on s3]",
#     "weights": [
#         {
#             "name": "Mistral-7B-v0.3",
#             "value": "mistralai/Mistral-7B-v0.3",
#             "size": 37,
#             "parameters": "7b",
#             "tflops": 15,
#             "vram": 20,
#             "nodes": 1,
#         },
#         {
#             "name": "Mistral-7B-v0.2",
#             "value": "mistralai/Mistral-7B-v0.2",
#             "size": 37,
#             "parameters": "7b",
#             "tflops": 15,
#             "vram": 20,
#             "nodes": 1,
#         },
#         {
#             "name": "Mistral-7B-v0.1",
#             "value": "mistralai/Mistral-7B-v0.1",
#             "size": 37,
#             "parameters": "7b",
#             "tflops": 15,
#             "vram": 20,
#             "nodes": 1,
#         },
#         {
#             "name": "Mistral-Nemo-Base-2407",
#             "value": "mistralai/Mistral-Nemo-Base-2407",
#             "size": 62,
#             "parameters": "12b",
#             "tflops": 32,
#             "vram": 32,
#             "nodes": 2,
#         },
#         {
#             "name": "Mistral-Small-24B-Instruct-2501",
#             "value": "mistralai/Mistral-Small-24B-Instruct-2501",
#             "size": 62,
#             "parameters": "24b",
#             "tflops": 64,
#             "vram": 64,
#             "nodes": 2,
#         },
#     ],
#     "cuda": "12.6",
#     "task": [
#         "text-generation",
#         "question-answering",
#         "image-text-to-text",
#     ],
# }
import json
import os
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from typing import Iterator

import gradio as gr
import spaces
import torch
from aixblock_ml.model import AIxBlockMLBase
from huggingface_hub import HfFolder, login
from loguru import logger
from mcp.server.fastmcp import FastMCP
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from function_ml import connect_project, download_dataset, upload_checkpoint
from logging_class import start_queue, write_log
from prompt import qa_without_context
import gc

# ------------------------------------------------------------------------------
hf_token = os.getenv("HF_TOKEN", "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")
HfFolder.save_token(hf_token)


hf_access_token = "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"
login(token=hf_access_token)
CUDA_VISIBLE_DEVICES = []
for i in range(torch.cuda.device_count()):
    CUDA_VISIBLE_DEVICES.append(i)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    f"{i}" for i in range(len(CUDA_VISIBLE_DEVICES))
)
print(os.environ["CUDA_VISIBLE_DEVICES"])


HOST_NAME = os.environ.get("HOST_NAME", "https://dev-us-west-1.aixblock.io")
TYPE_ENV = os.environ.get("TYPE_ENV", "DETECTION")


mcp = FastMCP("aixblock-mcp")

CHANNEL_STATUS = {}
# Parameters for model demo
model_demo = None
tokenizer_demo = None
model_loaded_demo = False
# Parameters for model deployment
pipe_prediction = None


class MyModel(AIxBlockMLBase):

    @mcp.tool()
    def action(self, command, **kwargs):
        """
        Execute commands for model operations, including shell command execution and model training.

        Args:
            command (str): The command to execute. Supported commands:
                - 'execute': Run a shell command
                - 'train': Start model training process
                - 'predict': Run model inference
                - 'tensorboard': Launch TensorBoard visualization
            **kwargs: Variable keyword arguments including:
                For 'execute' command:
                    - shell (str): The shell command to execute
                For 'train' command:
                    - model_id (str): Model identifier (default: 'Qwen/Qwen2.5-Coder-7B-Instruct')
                    - dataset_id (str): Dataset identifier
                    - push_to_hub (bool): Whether to push to HuggingFace Hub (default: True)
                    - hf_model_id (str): HuggingFace model ID
                    - push_to_hub_token (str): HuggingFace authentication token
                    - framework (str): Training framework (default: 'huggingface')
                    - task (str): Training task type (default: 'text-generation')
                    - trainingArguments (dict): Training configuration parameters
                    - cuda_debug (bool): Enable CUDA debugging (default: False)
                For 'predict' command:
                    - input_text (str): Text input for inference
                    - max_length (int): Maximum length of generated text (default: 512)
                    - temperature (float): Sampling temperature (default: 0.7)
                For 'tensorboard' command:
                    - logdir (str): Directory containing TensorBoard logs
                    - port (int): Port to run TensorBoard server (default: 6006)

        Returns:
            dict: A dictionary containing operation status or results
        """
        logger.info(f"Received command: {command} with args: {kwargs}")
        if command.lower() == "execute":
            _command = kwargs.get("shell", None)
            logger.info(f"Executing command: {_command}")
            subprocess.Popen(
                _command,
                shell=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True,
            )
            return {"message": "command completed successfully"}

        elif command.lower() == "train":

            model_id = kwargs.get("model_id", "mistralai/Mistral-7B-Instruct-v0.2")
            dataset_id = kwargs.get(
                "dataset_id", "autoprogrammer/Qwen2.5-Coder-7B-Instruct-codeguardplus"
            )

            push_to_hub = kwargs.get("push_to_hub", True)
            hf_model_id = kwargs.get(
                "hf_model_id", "mistralai/Mistral-7B-Instruct-v0.2"
            )
            push_to_hub_token = kwargs.get(
                "push_to_hub_token", "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"
            )
            framework = kwargs.get("framework", "huggingface")
            task = kwargs.get("task", "text-generation")
            prompt = kwargs.get("prompt", "")
            trainingArguments = kwargs.get("TrainingArguments", None)
            cuda_debug = kwargs.get("cuda_debug", False)

            json_file = "training_args.json"
            absolute_path = os.path.abspath(json_file)

            with open(absolute_path, "w") as f:
                json.dump(trainingArguments, f)
            logger.info(f"Training arguments: {trainingArguments}")

            if cuda_debug == True:
                os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
                os.environ["NCCL_DEBUG"] = "INFO"

            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
            os.environ["TORCH_USE_CUDA_DSA"] = "0"
            clone_dir = os.path.join(os.getcwd())
            project_id = kwargs.get("project_id", 0)
            token = kwargs.get("token", "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")
            checkpoint_version = kwargs.get("checkpoint_version")
            checkpoint_id = kwargs.get("checkpoint")
            dataset_version = kwargs.get("dataset_version")
            dataset = kwargs.get("dataset")
            channel_log = kwargs.get("channel_log", "training_logs")
            world_size = kwargs.get("world_size", 1)
            rank = kwargs.get("rank", 0)
            master_add = kwargs.get("master_add", "127.0.0.1")
            master_port = kwargs.get("master_port", "23456")
            host_name = kwargs.get("host_name", HOST_NAME)
            instruction_field = kwargs.get("prompt_field", "prompt")
            input_field = kwargs.get("input_field", "task_description")
            output_field = kwargs.get("output_field", "response")
            log_queue, logging_thread = start_queue(channel_log)
            write_log(log_queue)
            channel_name = f"{hf_model_id}_{str(uuid.uuid4())[:8]}"
            username = ""
            hf_model_name = ""
            CHANNEL_STATUS[channel_name] = {
                "status": "training",
                "hf_model_id": hf_model_name,
                "command": command,
                "created_at": time.time(),
            }
            print(f"🚀 Đã bắt đầu training kênh: {channel_name}")

            def func_train_model(
                clone_dir,
                project_id,
                token,
                checkpoint_version,
                checkpoint_id,
                dataset_version,
                dataset_id,
                model_id,
                world_size,
                rank,
                master_add,
                master_port,
                prompt,
                json_file,
                channel_log,
                hf_model_id,
                push_to_hub,
                push_to_hub_token,
                host_name,
            ):

                dataset_path = None
                project = connect_project(host_name, token, project_id)

                if dataset_version and dataset_id and project:
                    dataset_path = os.path.join(
                        clone_dir, f"datasets/{dataset_version}"
                    )

                    if not os.path.exists(dataset_path):
                        data_path = os.path.join(clone_dir, "data_zip")
                        os.makedirs(data_path, exist_ok=True)

                        dataset_name = download_dataset(project, dataset_id, data_path)
                        print(dataset_name)
                        if dataset_name:
                            data_zip_dir = os.path.join(data_path, dataset_name)

                            with zipfile.ZipFile(data_zip_dir, "r") as zip_ref:
                                zip_ref.extractall(dataset_path)

                            extracted_files = os.listdir(dataset_path)
                            zip_files = [
                                f for f in extracted_files if f.endswith(".zip")
                            ]

                            if len(zip_files) == 1:
                                inner_zip_path = os.path.join(
                                    dataset_path, zip_files[0]
                                )
                                print(
                                    f"🔁 Found inner zip file: {inner_zip_path}, extracting..."
                                )
                                with zipfile.ZipFile(inner_zip_path, "r") as inner_zip:
                                    inner_zip.extractall(dataset_path)
                                os.remove(inner_zip_path)

                subprocess.run(
                    ("whereis accelerate"),
                    shell=True,
                )
                print("===Train===")
                if framework == "huggingface":
                    if int(world_size) > 1:
                        if int(rank) == 0:
                            print("master node")
                            command = (
                                "venv/bin/accelerate launch --num_processes {num_processes} --num_machines {SLURM_NNODES} --machine_rank 0 --main_process_ip {head_node_ip} --main_process_port {port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field}"
                            ).format(
                                num_processes=world_size * torch.cuda.device_count(),
                                SLURM_NNODES=world_size,
                                head_node_ip=master_add,
                                port=master_port,
                                file_name="./run_distributed_accelerate.py",
                                json_file=json_file,
                                dataset_path=dataset_path,
                                channel_log=channel_log,
                                hf_model_id=hf_model_id,
                                push_to_hub=push_to_hub,
                                model_id=model_id,
                                push_to_hub_token=push_to_hub_token,
                                instruction_field=instruction_field,
                                input_field=input_field,
                                output_field=output_field,
                            )
                            process = subprocess.run(
                                command,
                                shell=True,
                            )
                        else:
                            print("worker node")
                            command = (
                                "venv/bin/accelerate launch --num_processes {num_processes} --num_machines {SLURM_NNODES} --machine_rank {machine_rank} --main_process_ip {head_node_ip} --main_process_port {port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field}"
                            ).format(
                                num_processes=world_size * torch.cuda.device_count(),
                                SLURM_NNODES=world_size,
                                head_node_ip=master_add,
                                port=master_port,
                                machine_rank=rank,
                                file_name="./run_distributed_accelerate.py",
                                json_file=json_file,
                                dataset_path=dataset_path,
                                channel_log=channel_log,
                                hf_model_id=hf_model_id,
                                push_to_hub=push_to_hub,
                                model_id=model_id,
                                push_to_hub_token=push_to_hub_token,
                                instruction_field=instruction_field,
                                input_field=input_field,
                                output_field=output_field,
                            )
                            process = subprocess.run(
                                command,
                                shell=True,
                            )

                    else:
                        if torch.cuda.device_count() > 1:  # multi gpu
                            command = (
                                "venv/bin/accelerate launch --multi_gpu --num_machines {SLURM_NNODES} --machine_rank 0 --num_processes {num_processes} {file_name} --training_args_json {json_file}  --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field}"
                            ).format(
                                num_processes=world_size * torch.cuda.device_count(),
                                SLURM_NNODES=world_size,
                                # head_node_ip=os.environ.get("head_node_ip", master_add),
                                port=master_port,
                                file_name="./run_distributed_accelerate.py",
                                json_file=json_file,
                                dataset_path=dataset_path,
                                channel_log=channel_log,
                                hf_model_id=hf_model_id,
                                push_to_hub=push_to_hub,
                                model_id=model_id,
                                push_to_hub_token=push_to_hub_token,
                                instruction_field=instruction_field,
                                input_field=input_field,
                                output_field=output_field,
                            )
                            print("================2")
                            print(command)
                            print("================2")
                            process = subprocess.run(command, shell=True)

                        elif torch.cuda.device_count() == 1:  # one gpu
                            command = (
                                "venv/bin/accelerate launch {file_name} --training_args_json {json_file}  --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field}"
                            ).format(
                                file_name="./run_distributed_accelerate.py",
                                json_file=json_file,
                                dataset_path=dataset_path,
                                channel_log=channel_log,
                                hf_model_id=hf_model_id,
                                push_to_hub={push_to_hub},
                                model_id=model_id,
                                push_to_hub_token={push_to_hub_token},
                                instruction_field=instruction_field,
                                input_field=input_field,
                                output_field=output_field,
                            )
                            print("================")
                            print(command)
                            print("================")
                            process = subprocess.Popen(
                                command,
                                shell=True,
                                stdout=sys.stdout,
                                stderr=sys.stderr,
                                text=True,
                            )
                        else:  # no gpu
                            command = (
                                "venv/bin/accelerate launch --cpu {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field}"
                            ).format(
                                file_name="./run_distributed_accelerate.py",
                                json_file=json_file,
                                dataset_path=dataset_path,
                                channel_log=channel_log,
                                hf_model_id=hf_model_id,
                                push_to_hub=push_to_hub,
                                model_id=model_id,
                                push_to_hub_token=push_to_hub_token,
                                instruction_field=instruction_field,
                                input_field=input_field,
                                output_field=output_field,
                            )
                            process = subprocess.Popen(
                                command,
                                shell=True,
                                stdout=sys.stdout,
                                stderr=sys.stderr,
                                text=True,
                            )
                            while True:
                                output = process.stdout.readline().decode("utf-8")
                                if output == "" and process.poll() is not None:
                                    break
                                if output:
                                    print(output, end="")
                            process.wait()

                elif framework == "pytorch":
                    process = subprocess.run(
                        ("whereis torchrun"),
                        shell=True,
                    )

                    if int(world_size) > 1:
                        if rank == 0:
                            print("master node")
                            command = (
                                "venv/bin/torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                                "--master_addr {master_addr} --master_port {master_port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field}"
                            ).format(
                                nnodes=int(world_size),
                                node_rank=int(rank),
                                nproc_per_node=world_size * torch.cuda.device_count(),
                                master_addr="127.0.0.1",
                                master_port="23456",
                                file_name="./run_distributed_accelerate.py",
                                json_file=json_file,
                                dataset_path=dataset_path,
                                channel_log=channel_log,
                                hf_model_id=hf_model_id,
                                push_to_hub=push_to_hub,
                                model_id=model_id,
                                push_to_hub_token=push_to_hub_token,
                                instruction_field=instruction_field,
                                input_field=input_field,
                                output_field=output_field,
                            )
                            process = subprocess.Popen(
                                command,
                                shell=True,
                                stdout=sys.stdout,
                                stderr=sys.stderr,
                                text=True,
                            )
                        else:
                            print("worker node")
                            command = (
                                "venv/bin/torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                                "--master_addr {master_addr} --master_port {master_port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field}"
                            ).format(
                                nnodes=int(world_size),
                                node_rank=int(rank),
                                nproc_per_node=world_size * torch.cuda.device_count(),
                                master_addr=master_add,
                                master_port=master_port,
                                file_name="./run_distributed_accelerate.py",
                                json_file=json_file,
                                dataset_path=dataset_path,
                                channel_log=channel_log,
                                hf_model_id=hf_model_id,
                                push_to_hub=push_to_hub,
                                model_id=model_id,
                                push_to_hub_token=push_to_hub_token,
                                instruction_field=instruction_field,
                                input_field=input_field,
                                output_field=output_field,
                            )
                            print(command)
                            process = subprocess.Popen(
                                command,
                                shell=True,
                                stdout=sys.stdout,
                                stderr=sys.stderr,
                                text=True,
                            )
                    else:
                        command = (
                            "venv/bin/torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                            "{file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field}"
                        ).format(
                            nnodes=int(world_size),
                            node_rank=int(rank),
                            nproc_per_node=world_size * torch.cuda.device_count(),
                            file_name="./run_distributed_accelerate.py",
                            json_file=json_file,
                            dataset_path=dataset_path,
                            channel_log=channel_log,
                            hf_model_id=hf_model_id,
                            push_to_hub=push_to_hub,
                            model_id=model_id,
                            push_to_hub_token=push_to_hub_token,
                            instruction_field=instruction_field,
                            input_field=input_field,
                            output_field=output_field,
                        )
                        process = subprocess.run(
                            command,
                            shell=True,
                        )
                CHANNEL_STATUS[channel_name]["status"] = "done"
                output_dir = "./data/checkpoint"
                print(push_to_hub)
                if push_to_hub:
                    import datetime

                    output_dir = "./data/checkpoint"
                    now = datetime.datetime.now()
                    date_str = now.strftime("%Y%m%d")
                    time_str = now.strftime("%H%M%S")
                    version = f"{date_str}-{time_str}"

                    upload_checkpoint(project, version, output_dir)

            train_thread = threading.Thread(
                target=func_train_model,
                args=(
                    clone_dir,
                    project_id,
                    token,
                    checkpoint_version,
                    checkpoint_id,
                    dataset_version,
                    dataset_id,
                    model_id,
                    world_size,
                    rank,
                    master_add,
                    master_port,
                    prompt,
                    absolute_path,
                    channel_log,
                    hf_model_id,
                    push_to_hub,
                    push_to_hub_token,
                    host_name,
                ),
            )
            train_thread.start()

            return {
                "message": "train completed successfully",
                "channel_name": channel_name,
            }
        elif command.lower() == "stop":
            subprocess.run(["pkill", "-9", "-f", "./inference/generate.py"])
            return {"message": "train stop successfully", "result": "Done"}

        elif command.lower() == "tensorboard":

            def run_tensorboard():
                p = subprocess.Popen(
                    f"tensorboard --logdir /app/data/checkpoint/runs --host 0.0.0.0 --port=6006",
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    text=True,
                )
                out = p.communicate()
                print(out)

            tensorboard_thread = threading.Thread(target=run_tensorboard)
            tensorboard_thread.start()
            return {"message": "tensorboardx started successfully"}

        elif command.lower() == "predict":
            prompt = kwargs.get("prompt", None)
            model_id = kwargs.get("model_id", "mistralai/Mistral-7B-Instruct-v0.2")
            text = kwargs.get("text", None)
            token_length = kwargs.get("token_lenght", 30)
            task = kwargs.get("task", "")
            voice = kwargs.get("voice", "")
            max_new_token = kwargs.get("max_new_token", 256)
            temperature = kwargs.get("temperature", 0.7)
            top_k = kwargs.get("top_k", 50)
            top_p = kwargs.get("top_p", 0.95)

            predictions = []

            if not prompt or prompt == "":
                prompt = text

            from huggingface_hub import login

            hf_access_token = kwargs.get(
                "hf_access_token", "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"
            )
            login(token=hf_access_token)

            def smart_pipeline(
                model_id: str,
                token: str,
                local_dir="./data/checkpoint",
                task="text-generation",
            ):
                global pipe_prediction

                if pipe_prediction == None:
                    try:
                        model_name = model_id.split("/")[-1]
                        local_model_dir = os.path.join(local_dir, model_name)
                        if os.path.exists(local_model_dir) and os.path.exists(
                            os.path.join(local_model_dir, "config.json")
                        ):
                            print(f"✅ Loading model from local: {local_model_dir}")
                            model_source = local_model_dir
                        else:
                            print(f"☁️ Loading model from HuggingFace Hub: {model_id}")
                            model_source = model_id
                    except:
                        print(f"☁️ Loading model from HuggingFace Hub: {model_id}")
                        model_source = model_id

                    if torch.cuda.is_available():
                        if torch.cuda.is_bf16_supported():
                            dtype = torch.bfloat16
                        else:
                            dtype = torch.float16

                        print("Using CUDA.")
                        pipe_prediction = pipeline(
                            task,
                            model=model_source,
                            torch_dtype=dtype,
                            device_map="auto",
                            token=token,
                            max_new_tokens=256,
                        )
                    else:
                        print("Using CPU.")
                        pipe_prediction = pipeline(
                            task,
                            model=model_source,
                            device_map="cpu",
                            token=token,
                            max_new_tokens=256,
                        )

            with torch.no_grad():
                # Load the model
                smart_pipeline(model_id, hf_access_token)
                generated_text = qa_without_context(pipe_prediction, prompt)

            print(generated_text)
            predictions.append(
                {
                    "result": [
                        {
                            "from_name": "generated_text",
                            "to_name": "text_output",
                            "type": "textarea",
                            "value": {"text": [generated_text]},
                        }
                    ],
                    "model_version": "",
                }
            )

            return {"message": "predict completed successfully", "result": predictions}
        elif command.lower() == "prompt_sample":
            task = kwargs.get("task", "")
            if task == "question-answering":
                prompt_text = f"""
                    Here is the context: 
                    {{context}}

                    Based on the above context, provide an answer to the following question: 
                    {{question}}

                    Answer:
                    """
            elif task == "text-classification":
                prompt_text = f"""
                    Summarize the following text into a single, concise paragraph focusing on the key ideas and important points:

                    Text: 
                    {{context}}

                    Summary:
                    """

            elif task == "summarization":
                prompt_text = f"""
                    Summarize the following text into a single, concise paragraph focusing on the key ideas and important points:

                    Text: 
                    {{context}}

                    Summary:
                    """
            return {
                "message": "prompt_sample completed successfully",
                "result": prompt_text,
            }

        elif command.lower() == "action-example":
            return {"message": "Done", "result": "Done"}

        elif command == "status":
            channel = kwargs.get("channel", None)

            if channel:
                # Nếu có truyền kênh cụ thể
                status_info = CHANNEL_STATUS.get(channel)
                if status_info is None:
                    return {"channel": channel, "status": "not_found"}
                elif isinstance(status_info, dict):
                    return {"channel": channel, **status_info}
                else:
                    return {"channel": channel, "status": status_info}
            else:
                # Lấy tất cả kênh
                if not CHANNEL_STATUS:
                    return {"message": "No channels available"}

                channels = []
                for ch, info in CHANNEL_STATUS.items():
                    if isinstance(info, dict):
                        channels.append({"channel": ch, **info})
                    else:
                        channels.append({"channel": ch, "status": info})

                return {"channels": channels}
        else:
            return {"message": "command not supported", "result": None}

    @mcp.tool()
    def model(self, **kwargs):
        """
        This tool demos a code model with 7B parameters fine-tuned for chat instructions.
        You can interact with the model by sending a message and the model will generate a response based on the input.
        The model is loaded from huggingface hub and can be customized by passing the model name and other parameters.
        For example, you can try the 7B model in the official homepage.

        Args:
            model_id (str, optional): The model id to load from huggingface hub. Defaults to "Qwen/Qwen2.5-Coder-7B-Instruct".
            project_id (int, optional): The project id to use for the gradio app. Defaults to 0.
            hf_access_token (str, optional): The huggingface access token to use for loading the model. Defaults to "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".

        Returns:
            dict: A dictionary containing the share url and local url of the gradio app.
        """
        global model_demo, tokenizer_demo, model_loaded_demo, model_id_demo

        model_id_demo = kwargs.get("model_id", "mistralai/Mistral-7B-Instruct-v0.2")
        project_id = kwargs.get("project_id", 0)

        print(
            f"""\
        Project ID: {project_id}
        """
        )
        from huggingface_hub import login

        hf_access_token = kwargs.get(
            "hf_access_token", "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"
        )
        login(token=hf_access_token)
        MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

        DESCRIPTION = """\
        # Mistral
        """

        if not torch.cuda.is_available():
            DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        def load_model(model_id):
            global model_demo, tokenizer_demo, model_loaded_demo
            if torch.cuda.is_available() and not model_loaded_demo:
                model_demo = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    token=hf_access_token,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                    torch_dtype=compute_dtype,
                )
                tokenizer_demo = AutoTokenizer.from_pretrained(
                    model_id, token=hf_access_token
                )
                tokenizer_demo.use_default_system_prompt = False
                model_loaded_demo = True
                return f"Model {model_id} loaded successfully!"
            elif model_loaded_demo:
                return "Model is already loaded! Please refresh the page to load a different model."
            else:
                return "Error: CUDA is not available!"

        @spaces.GPU
        def generate(
            message: str,
            chat_history: list[tuple[str, str]],
            system_prompt: str,
            max_new_tokens: int = 1024,
            temperature: float = 0.6,
            top_p: float = 0.9,
            top_k: int = 50,
            repetition_penalty: float = 1,
        ) -> Iterator[str]:
            if not model_loaded_demo:
                return (
                    "Please load the model first by clicking the 'Load Model' button."
                )
            chat_messages = []
            if system_prompt:
                chat_messages.append({"role": "system", "content": str(system_prompt)})

            # Add chat history
            for user_msg, assistant_msg in chat_history:
                chat_messages.append({"role": "user", "content": str(user_msg)})
                chat_messages.append(
                    {"role": "assistant", "content": str(assistant_msg)}
                )

            # Add the current message
            chat_messages.append({"role": "user", "content": str(message)})
            text = tokenizer_demo.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = tokenizer_demo([text], return_tensors="pt").to(
                model_demo.device
            )
            if model_inputs.input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
                model_inputs.input_ids = model_inputs.input_ids[
                    :, -MAX_INPUT_TOKEN_LENGTH:
                ]
                gr.Warning(
                    f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens."
                )

            generated_ids = model_demo.generate(**model_inputs, max_new_tokens=512)

            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer_demo.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            return response

        chat_interface = gr.ChatInterface(
            fn=generate,
            stop_btn=gr.Button("Stop"),
            examples=[
                ["implement snake game using pygame"],
                [
                    "Can you explain briefly to me what is the Python programming language?"
                ],
                ["write a program to find the factorial of a number"],
            ],
        )

        with gr.Blocks(css="style.css") as demo:
            gr.Markdown(DESCRIPTION)
            with gr.Row():
                load_btn = gr.Button("Load Model")
                status_text = gr.Textbox(label="Model Status", interactive=False)
            load_btn.click(fn=lambda: load_model(model_id_demo), outputs=status_text)
            chat_interface.render()

        gradio_app, local_url, share_url = demo.launch(
            share=True,
            quiet=True,
            prevent_thread_lock=True,
            server_name="0.0.0.0",
            show_error=True,
        )
        return {"share_url": share_url, "local_url": local_url}

    @mcp.tool()
    def model_trial(self, project, **kwargs):
        return {"message": "Done", "result": "Done"}

    @mcp.tool()
    def download(self, project, **kwargs):
        from flask import request, send_from_directory

        file_path = request.args.get("path")
        print(request.args)
        return send_from_directory(os.getcwd(), file_path)
