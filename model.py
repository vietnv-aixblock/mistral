#model_marketplace.config
# {"token_length": "4018", "accuracy": "70", "precision": "fp16", "sampling_frequency:": "44100", "mono": true, "fps": "74", "resolution": "480", "image_width": "1080", "image_height": "1920", "framework": "transformers", "dataset_format": "llm", "dataset_sample": "[id on s3]", "weights": [
#     {
#       "name": "DeepSeek-V3",
#       "value": "deepseek-ai/DeepSeek-V3",
#       "size": 20,
#       "paramasters": "685B",
#       "tflops": 14, 
#       "vram": 20,
#       "nodes": 10
#     },
# {
#       "name": "DeepSeek-V3-bf16",
#       "value": "opensourcerelease/DeepSeek-V3-bf16",
#       "size": 1500,
#       "paramasters": "684B",
#       "tflops": 80, 
#       "vram": 48,
#       "nodes": 10
#     }
#   ], "cuda": "11.4", "task":["text-generation", "text-classification", "text-summarization", "text-ner", "question-answering"]}
import math
import pathlib
import random
import subprocess
import time
from typing import List, Dict, Optional
import accelerate
from aixblock_ml.model import AIxBlockMLBase
import torch
from transformers import pipeline
import os
from huggingface_hub import HfFolder
import wandb
from prompt import qa_with_context, text_classification, text_summarization, qa_without_context, text_ner, chatbot_with_history

# Đặt token của bạn vào đây
hf_token = os.getenv("HF_TOKEN", "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")
# Lưu token vào local
HfFolder.save_token(hf_token)
# wandb.login('allow',"69b9681e7dc41d211e8c93a3ba9a6fb8d781404a")
# print("Login successful")
from huggingface_hub import login 
hf_access_token = "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"
login(token = hf_access_token)

if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    print("CUDA is available.")
    
    _model = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-v0.1", #"meta-llama/Llama-3.2-1B-Instruct", #"meta-llama/Llama-3.2-3B", meta-llama/Llama-3.3-70B-Instruct
        torch_dtype=dtype, 
        device_map="auto",  # Hoặc có thể thử "cpu" nếu không ổn,
        max_new_tokens=256,
        token = "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"
    )
else:
    print("No GPU available, using CPU.")
    _model = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-v0.1", #"meta-llama/Llama-3.2-1B-Instruct", #"meta-llama/Llama-3.2-3B", meta-llama/Llama-3.3-70B-Instruct
        device_map="cpu",
        max_new_tokens=256,
        token = "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"
    )


from typing import List, Dict, Optional
from aixblock_ml.model import AIxBlockMLBase
import torch.distributed as dist
import os
import torch
import os
import subprocess
import random
import asyncio
import logging
import logging
import subprocess
import threading
import time
import base64
import json
import hashlib
import hmac
from dashboard import promethus_grafana
from centrifuge import CentrifugeError, Client, ClientEventHandler, SubscriptionEventHandler
def base64url_encode(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=")
def generate_jwt(user, channel=""):
    """Note, in tests we generate token on client-side - this is INSECURE
    and should not be used in production. Tokens must be generated on server-side."""
    hmac_secret = "d0a70289-9806-41f6-be6d-f4de5fe298fb"  # noqa: S105 - this is just a secret used in tests.
    header = {"typ": "JWT", "alg": "HS256"}
    payload = {"sub": user}
    if channel:
        # Subscription token
        payload["channel"] = channel
    encoded_header = base64url_encode(json.dumps(header).encode("utf-8"))
    encoded_payload = base64url_encode(json.dumps(payload).encode("utf-8"))
    signature_base = encoded_header + b"." + encoded_payload
    signature = hmac.new(hmac_secret.encode("utf-8"), signature_base, hashlib.sha256).digest()
    encoded_signature = base64url_encode(signature)
    jwt_token = encoded_header + b"." + encoded_payload + b"." + encoded_signature
    return jwt_token.decode("utf-8")
async def get_client_token() -> str:
    return generate_jwt("42")

async def get_subscription_token(channel: str) -> str:
    return generate_jwt("42", channel)

class ClientEventLoggerHandler(ClientEventHandler):
    async def on_connected(self, ctx):
        logging.info("Connected to server")

class SubscriptionEventLoggerHandler(SubscriptionEventHandler):
    async def on_subscribed(self, ctx):
        logging.info("Subscribed to channel")

def setup_client(channel_log):
    client = Client(
        "wss://rt.aixblock.io/centrifugo/connection/websocket",
        events=ClientEventLoggerHandler(),
        get_token=get_client_token,
        use_protobuf=False,
    )

    sub = client.new_subscription(
        channel_log,
        events=SubscriptionEventLoggerHandler(),
        # get_token=get_subscription_token,
    )

    return client, sub

async def send_log(sub, log_message):
    try:
        await sub.publish(data={"log": log_message})
    except CentrifugeError as e:
        logging.error("Error publish: %s", e)
async def send_message(sub, message):
    try:
        await sub.publish(data=message)
    except CentrifugeError as e:
        logging.error("Error publish: %s", e)
# def log_training_progress(log_message):
#     client.disconnect()  # Đóng kết nối

async def log_training_progress(sub, log_message):
    await send_log(sub, log_message)

def run_train(command, channel_log="training_logs"):
    def run():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            client, sub = setup_client(channel_log)
            async def main():
                await client.connect()
                await sub.subscribe()
                await log_training_progress(sub, "Training started")
                log_file_path = "logs/llm-ddp.log"
                last_position = 0  # Vị trí đã đọc đến trong file log
                await log_training_progress(sub, "Training training")
                promethus_grafana.promethus_push_to("training")
                
                while True:
                    try:
                        current_size = os.path.getsize(log_file_path)
                        if current_size > last_position:
                            with open(log_file_path, "r") as log_file:
                                log_file.seek(last_position)
                                new_lines = log_file.readlines()
                                # print(new_lines)
                                for line in new_lines:
                                    print("--------------" ,f"{line.strip()}")
                        #             # Thay thế đoạn này bằng code để gửi log
                                    await log_training_progress(sub, f"{line.strip()}")
                            last_position = current_size  

                        time.sleep(5)
                    except Exception as e:
                        print(e)

                # promethus_grafana.promethus_push_to("finish")
                # await log_training_progress(sub, "Training completed")
                # await client.disconnect()
                # loop.stop()

            try:
                loop.run_until_complete(main())
            finally:
                loop.close()  # Đảm bảo vòng lặp được đóng lại hoàn toàn
    
        except Exception as e:
            print(e)

    thread_start = threading.Thread(target=run)
    thread_start.start()
    subprocess.run(command, shell=True)
    # try:
    #     promethus_grafana.promethus_push_to("finish")
    # except:
    #     pass

def fetch_logs(channel_log="training_logs"):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    client, sub = setup_client(channel_log)

    async def run():
        await client.connect()
        await sub.subscribe()
        history = await sub.history(limit=-1)
        logs = []
        for pub in history.publications:
            log_message = pub.data.get('log')
            if log_message:
                logs.append(log_message)
        await client.disconnect()
        return logs

    return loop.run_until_complete(run())
class MyModel(AIxBlockMLBase):

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """ 
        """
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}''')
        return []

    def fit(self, event, data, **kwargs):
        """

        
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')
    def action(self, project, command, collection, **kwargs):
        
        print(f"""
              project: {project},
                command: {command},
                collection: {collection},
              """)
        if command.lower() == "train":
                import threading
                import os

                model_id = kwargs.get("model_id", "mistralai/Mistral-7B-Instruct-v0.3")  #"tiiuae/falcon-7b" "bigscience/bloomz-1b7" `zanchat/falcon-1b` `appvoid/llama-3-1b` meta-llama/Llama-3.2-3B` `mistralai/Mistral-7B-v0.1` `bigscience/bloomz-1b7` `Qwen/Qwen2-1.5B`
                dataset_id = kwargs.get("dataset_id","lucasmccabe-lmi/CodeAlpaca-20k") #gingdev/llama_vi_52k kigner/ruozhiba-llama3-tt

                push_to_hub = kwargs.get("push_to_hub", True)
                hf_model_id = kwargs.get("hf_model_id", "tonyshark/llama3")
                
                task = kwargs.get("task", "text-generation")
                framework = kwargs.get("framework", "huggingface")
                prompt = kwargs.get("prompt", "")
                report_to = kwargs.get("report_to", "tensorboard")
                wantdb_api_key = kwargs.get("wantdb_api_key", "69b9681e7dc41d211e8c93a3ba9a6fb8d781404a")
                trainingArguments = kwargs.get("TrainingArguments", None)

                json_file = "training_args.json"
                absolute_path = os.path.abspath(json_file)

                with open(absolute_path, 'w') as f:
                    json.dump(trainingArguments, f)

                print(trainingArguments)
                if len(wantdb_api_key)> 0 and wantdb_api_key != "69b9681e7dc41d211e8c93a3ba9a6fb8d781404a":
                    wandb.login('allow',wantdb_api_key)

                os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
                clone_dir = os.path.join(os.getcwd())
                # epochs = kwargs.get("num_epochs", 10)
                project_id = kwargs.get("project_id")
                token = kwargs.get("token")
                checkpoint_version = kwargs.get("checkpoint_version")
                checkpoint_id = kwargs.get("checkpoint")
                dataset_version = kwargs.get("dataset_version")
                # dataset_id = kwargs.get("dataset")
                channel_log = kwargs.get("channel_log", "training_logs")
                world_size = kwargs.get("world_size", 1)
                rank = kwargs.get("rank", 0)
                master_add = kwargs.get("master_add","127.0.0.1")
                master_port = kwargs.get("master_port", "12345")
                # entry_file = kwargs.get("entry_file")
                configs = kwargs.get("configs")
                # hyperparameter = kwargs.get("hyperparameter")

                # !pip install nvidia-ml-py
                # from pynvml import *
                # nvmlInit()
                # print(f"Driver Version: {nvmlSystemGetDriverVersion()}")
                # # Driver Version: 11.515.48
                # deviceCount = nvmlDeviceGetCount()
                # for i in range(deviceCount):
                # handle = nvmlDeviceGetHandleByIndex(i)
                # print(f"Device {i} : {nvmlDeviceGetName(handle)}")
                # nvmlShutdown()

                def func_train_model(clone_dir, project_id, token, checkpoint_version, checkpoint_id, dataset_version, dataset_id,model_id,world_size,rank,master_add,master_port,prompt, json_file):
                    from misc import get_device_count
                    # https://huggingface.co/docs/accelerate/en/basic_tutorials/launch
                    # https://huggingface.co/docs/accelerate/en/package_reference/cli
                    subprocess.run(
                                (
                                    "whereis accelerate"
                                ),
                                shell=True,
                            )
                    if framework == "huggingface":
                        if world_size > 1:
                            if rank == 0:
                                print("master node")
                                #  --dynamo_backend 'no' \
                                command = (
                                        "accelerate launch --num_processes {num_processes} --num_machines {SLURM_NNODES} --rdzv_backend c10d --main_process_ip {head_node_ip} --main_process_port {port} {file_name} --training_args_json {json_file}"
                                    ).format(
                                        num_processes=world_size*get_device_count(),
                                        SLURM_NNODES=world_size,
                                        head_node_ip=master_add,
                                        port=master_port,
                                        file_name="./run_distributed_accelerate.py", 
                                        json_file=json_file
                                    )
                                # process = subprocess.run(
                                #     command,
                                #     shell=True,
                                # )
                                run_train(command)
                            else:
                                print("worker node")
                                command = (
                                        "accelerate launch --num_processes {num_processes} --num_machines {SLURM_NNODES} --rdzv_backend c10d --main_process_ip {head_node_ip} --main_process_port {port} {file_name} --training_args_json {json_file}"
                                    ).format(
                                        num_processes=world_size*get_device_count(),
                                        SLURM_NNODES=world_size,
                                        head_node_ip=master_add,
                                        port=master_port,
                                        file_name="./run_distributed_accelerate.py", 
                                        json_file=json_file
                                    )
                                num_processes = world_size*get_device_count()
                                # process = subprocess.run(
                                #     command,
                                #     shell=True,
                                # )
                                run_train(command)
                        else:
                            if get_device_count() > 1: # multi gpu
                                            #                                     --rdzv_backend c10d \
                                            # --main_process_ip {head_node_ip} \
                                            # --main_process_port {port} \
                                            # --mixed_precision 'fp16' \
                                            # --num_machines {SLURM_NNODES} \
                                command = (
                                        "accelerate launch --multi_gpu --main_process_port {port}  --num_processes {num_processes} {file_name} --training_args_json {json_file}"
                                    ).format(
                                        num_processes=world_size*get_device_count(),
                                        # SLURM_NNODES=os.environ.get("SLURM_NNODES", world_size),
                                        # head_node_ip=os.environ.get("head_node_ip", master_add),
                                        port=master_port,
                                            file_name="./run_distributed_accelerate.py", 
                                            json_file=json_file
                                    )
                                # process = subprocess.run(
                                #     command,
                                #     shell=True,
                                # )
                                run_train(command)
                                # mixed_precision: fp16
                            elif get_device_count() == 1: # one gpu
                                # num_processes = world_size*get_device_count()
                                #    --rdzv_backend c10d \
                                #     --main_process_ip {head_node_ip} \
                                #     --main_process_port {port} \
                                # --num_cpu_threads_per_process=2 --num_processes {num_processes} \
                                #     --num_machines {SLURM_NNODES} \
                                command = (
                                    "accelerate launch {file_name} --training_args_json {json_file}"
                                ).format(
                                    # num_processes=os.environ.get("num_processes", num_processes),
                                    # SLURM_NNODES=os.environ.get("SLURM_NNODES", world_size),
                                    # head_node_ip=os.environ.get("head_node_ip", master_add),
                                    # port=os.environ.get("port", master_port),
                                    file_name="./run_distributed_accelerate.py", 
                                    json_file=json_file
                                )
                                process = subprocess.run(
                                command,
                                shell=True,
                            )
                                run_train(command)
                                # python3.10 -m pip install -r requirements.txt
                            else: # no gpu
                                command = (
                                        "accelerate launch --cpu {file_name} --training_args_json {json_file}"
                                    ).format(
                                        file_name="./run_distributed_accelerate.py", 
                                        json_file=json_file
                                    )
                                
                                # process = subprocess.run(
                                #     command,
                                #     shell=True,
                                # )
                                run_train(command)

                    elif framework == "pytorch":
                        import torch
                        from peft import AutoPeftModelForCausalLM
                        from transformers import AutoTokenizer, pipeline
                        from datasets import load_dataset
                        from random import randint
                        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments #BitsAndBytesConfig,AutoConfig,
                        # from trl import SFTTrainer
                        import pathlib
                        from transformers import TrainerCallback
                        import numpy as np
                        from torch.utils.data.dataloader import DataLoader
                        from transformers import DataCollatorWithPadding      
                        process = subprocess.run(
                                (
                                    "whereis torchrun"
                                ),
                                shell=True,
                            )
                        # print(process.stdout.replace("torchrun: ",""))
                        if world_size > 1:
                            if rank == 0:
                                print("master node")
                                # args = f"--model_id {model_id} --dataset_id {dataset_id} --num_train_epochs {num_train_epochs} " \
                                #     f"--bf16 {bf16} --fp16 {fp16} --use_cpu {use_cpu} --push_to_hub {push_to_hub} " \
                                #     f"--hf_model_id {hf_model_id} --max_seq_length {max_seq_length} --framework {framework} " \
                                #     f"--per_device_train_batch_size {per_device_train_batch_size} --gradient_accumulation_steps {gradient_accumulation_steps} " \
                                #     f"--gradient_checkpointing {gradient_checkpointing} --optim {optim} --logging_steps {logging_steps} " \
                                #     f"--learning_rate {learning_rate} --max_grad_norm {max_grad_norm} --lora_alpha {lora_alpha} " \
                                #     f"--lora_dropout {lora_dropout} --bias {bias} --target_modules {target_modules} --task_type {task_type}"
                                #  --dynamo_backend 'no' \
                                command = (
                                        "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                                        "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
                                    ).format(
                                        nnodes=world_size,
                                        node_rank=rank,
                                        nproc_per_node=get_device_count(),
                                        master_addr="127.0.0.1",
                                        master_port="12345",
                                        file_name="test_accelerate.py",
                                        args="--model_id {model_id}"
                                    )
                                # process = subprocess.run(
                                #     command,
                                #     shell=True,
                                # )
                                run_train(command)
                            else:
                                print("worker node")
                                # args = f"--model_id {model_id} --dataset_id {dataset_id} --num_train_epochs {num_train_epochs} " \
                                #     f"--bf16 {bf16} --fp16 {fp16} --use_cpu {use_cpu} --push_to_hub {push_to_hub} " \
                                #     f"--hf_model_id {hf_model_id} --max_seq_length {max_seq_length} --framework {framework} " \
                                #     f"--per_device_train_batch_size {per_device_train_batch_size} --gradient_accumulation_steps {gradient_accumulation_steps} " \
                                #     f"--gradient_checkpointing {gradient_checkpointing} --optim {optim} --logging_steps {logging_steps} " \
                                #     f"--learning_rate {learning_rate} --max_grad_norm {max_grad_norm} --lora_alpha {lora_alpha} " \
                                #     f"--lora_dropout {lora_dropout} --bias {bias} --target_modules {target_modules} --task_type {task_type}"
                                command = (
                                        "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                                        "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
                                    ).format(
                                         nnodes=world_size,
                                        node_rank=rank,
                                        nproc_per_node=get_device_count(),
                                        master_addr=master_add,
                                        master_port=master_port,
                                        file_name="test_accelerate.py",
                                        args="--model_id {model_id}"
                                    )
                                print(command)
                                num_processes = world_size*get_device_count()
                                # process = subprocess.run(
                                #     command,
                                #     shell=True,
                                # )
                                run_train(command)
                        else:
                            command = (
                                        "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                                        "{file_name} {args}"
                                    ).format(
                                        nnodes=world_size,
                                        node_rank=rank,
                                        nproc_per_node=get_device_count(),
                                        # master_addr=master_add,
                                        # master_port=master_port,
                                        file_name="test_accelerate.py",
                                        args="" #" ".join(sys.argv[1:]),
                                    )
                            process = subprocess.run(
                                command,
                                shell=True,
                            )
                            run_train(command)
                    else:
                            import torch
                            from peft import AutoPeftModelForCausalLM
                            from transformers import AutoTokenizer, pipeline
                            from datasets import load_dataset
                            from random import randint
                            from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments #BitsAndBytesConfig,AutoConfig,
                            from trl import SFTTrainer 
                            import pathlib
                            from transformers import TrainerCallback
                            import numpy as np
                            from torch.utils.data.dataloader import DataLoader
                            from transformers import DataCollatorWithPadding
                            # Hugging Face model id
                            if model_id == None:
                                model_id = "mistralai/Mistral-7B-v0.1" # or  `appvoid/llama-3-1b` tiiuae/falcon-7b` `mistralai/Mistral-7B-v0.1` `bigscience/bloomz-1b7` `Qwen/Qwen2-1.5B`
                            train_dataset = load_dataset(dataset_id, split="train")
                            # train_valid = train_dataset['train'].train_test_split(test_size=0.2)
                            # test_valid = train_dataset['test'].train_test_split(test_size=0.2)
                            # train_dataset = train_valid['train']
                            # test_dataset = test_valid['test']
                            # print(f"train_dataset: {train_dataset}")
                            # print(f"test_dataset: {test_dataset}")
                    
                            alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                                            ### Instruction:
                                            {}

                                            ### Input:
                                            {}

                                            ### Response:
                                            {}"""
                            if len(prompt) > 0:
                                 alpaca_prompt = prompt
                        
                            # def formatting_prompts_func(examples):
                            #     instructions = examples["instruction"]
                            #     inputs       = examples["input"]
                            #     outputs      = examples["output"]
                            #     texts = []
                            #     for instruction, input, output in zip(instructions, inputs, outputs):
                            #         text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
                            #         texts.append(text)
                            #     return {"text": texts}
                            # def tokenizer_func(examples):
                            #     instructions = examples["instruction"]
                            #     inputs       = examples["input"]
                            #     outputs      = examples["output"]
                            #     texts = []
                            #     for instruction, input, output in zip(instructions, inputs, outputs):
                            #         text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
                            #         texts.append(text)
                            #     return tokenizer("".join(texts),truncation=True, padding=True, max_length=128, return_tensors="pt")
                            #                 # examples follow format of resp json files
                            # train_dataset = train_dataset.map(tokenizer_func,remove_columns=train_dataset.column_names, batched=True)
                            # Load model and tokenizer
                            model = AutoModelForCausalLM.from_pretrained(
                                model_id,
                                device_map="auto",
                                low_cpu_mem_usage=True,
                                use_cache=True
                            )
                            tokenizer = AutoTokenizer.from_pretrained(model_id)
                            EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
                            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                            tokenizer.pad_token = tokenizer.eos_token
                            def preprocess_function(examples):  
                                instructions = examples["instruction"]
                                inputs       = examples["input"]
                                outputs      = examples["output"]
                                texts = []
                                for instruction, input, output in zip(instructions, inputs, outputs):
                                    text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
                                    texts.append(text)
                                return tokenizer(texts,truncation=True, padding=True, max_length=128, return_tensors="pt")                         
                                # inputs = [ex['en'] for ex in examples['translation']]      
                                # targets = [ex['ru'] for ex in examples['translation']]
                                # model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, padding=True, truncation=True)
                            eval_dataset = train_dataset 
                            tokenized_datasets = train_dataset.map(
                                preprocess_function,
                                batched=True,
                                # remove_columns=train_dataset['train'].column_names,
                            )
                            # eval_tokenized_datasets = eval_dataset.map(
                            #     preprocess_function,
                            #     batched=True,
                            #     # remove_columns=train_dataset['train'].column_names,
                            # )

                            data_collator = DataCollatorWithPadding(tokenizer)
                            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16,collate_fn=data_collator)
                            eval_dataloader = DataLoader(eval_dataset, batch_size=16, collate_fn=data_collator)
                            # for epoch in range(2):
                            #     model.train()
                            #     for step, batch in enumerate(train_dataloader):          
                            #         outputs = model(**batch)
                            #         loss = outputs.loss
                            # print(len(train_dataset))
                            # print(len(eval_dataset))

                            with open(json_file, 'r') as f:
                                training_args_dict = json.load(f)

                            print("Data is formatted and ready!")
                            
                            # def compute_metrics(eval_pred):
                            #     predictions, labels = eval_pred
                            #     predictions = np.argmax(predictions, axis=1)
                            #     return accuracy.compute(predictions=predictions, references=labels)
                            
                            class TrainOnStartCallback(TrainerCallback):
                                def on_train_begin(self, args, state, control, logs=None, **kwargs):
                                    # Log training loss at step 0
                                    logs = logs or {}
                                    self.log(logs)

                                def log(self, logs):
                                    print(f"Logging at start: {logs}")

                            def is_valid_type(value, expected_type):
                                from typing import get_origin, get_args, Union
                                # Nếu không có type hint (Empty), chấp nhận giá trị
                                if expected_type is inspect._empty:
                                    return True
                                # Nếu type hint là generic (Union, Optional, List, etc.)
                                origin = get_origin(expected_type)
                                if origin is Union:  # Xử lý Union hoặc Optional
                                    return any(is_valid_type(value, arg) for arg in get_args(expected_type))
                                if origin is list:  # Xử lý List
                                    return isinstance(value, list) and all(is_valid_type(v, get_args(expected_type)[0]) for v in value)
                                if origin is dict:  # Xử lý Dict
                                    key_type, value_type = get_args(expected_type)
                                    return (
                                        isinstance(value, dict) and
                                        all(is_valid_type(k, key_type) for k in value.keys()) and
                                        all(is_valid_type(v, value_type) for v in value.values())
                                    )
                                # Kiểm tra kiểu cơ bản (int, float, str, etc.)
                                return isinstance(value, expected_type)


                            if training_args_dict:
                                import inspect
                                # Kết hợp dictionary từ JSON (nếu có) và giá trị mặc định
                                training_args_values = {**training_args_dict}

                                param_annotations = inspect.signature(TrainingArguments.__init__).parameters

                                valid_args = set(param_annotations.keys())
                                filtered_args = {k: v for k, v in training_args_values.items() if k in valid_args}

                                # Kiểm tra định dạng giá trị
                                validated_args = {}
                                for k, v in filtered_args.items():
                                    expected_type = param_annotations[k].annotation  # Lấy kiểu dữ liệu mong đợi
                                    if is_valid_type(v, expected_type):
                                        validated_args[k] = v
                                    else:
                                        print(f"Skipping invalid parameter: {k} (expected {expected_type}, got {type(v)})")

                                # Khởi tạo TrainingArguments với tham số đã được xác thực
                                training_args = TrainingArguments(**validated_args)

                            else:
                                training_args = TrainingArguments(
                                    output_dir= f"/app/data/checkpoint", # directory to save and repository id
                                    logging_dir= '/app/data/logs',
                                    learning_rate=2e-4,
                                    per_device_train_batch_size=3,
                                    per_device_eval_batch_size=16,
                                    num_train_epochs=10,
                                    weight_decay=0.01,
                                    save_strategy="epoch", 
                                    # report_to="tensorboard",
                                    report_to="wandb",
                                    use_cpu=False,
                                    bf16 = False,
                                    fp16 = False
                                )

                            
                            # https://github.com/huggingface/accelerate/issues/2618
                            # https://github.com/huggingface/huggingface-llama-recipes/blob/main/fine_tune/qlora_405B.slurm
                            # https://gist.github.com/rom1504/474f97a95a526d40ae44a3fc3c657a2e
                            # https://github.com/huggingface/accelerate/blob/main/examples/slurm/submit_multinode.sh
                            # https://github.com/huggingface/accelerate/blob/main/examples/slurm/submit_multigpu.sh
                            # https://github.com/huggingface/accelerate/blob/main/examples/slurm/submit_multinode_fsdp.sh
                            # https://github.com/huggingface/accelerate/blob/main/examples/slurm/submit_multicpu.sh
                            trainer = SFTTrainer(
                                dataset_text_field = "text",
                                model=model,
                                args=training_args,
                                train_dataset=tokenized_datasets,
                                # eval_dataset=eval_tokenized_datasets,
                                tokenizer=tokenizer,
                                # data_collator=data_collator,
                                # compute_metrics=compute_metrics,
                                dataset_kwargs={
                                                    "add_special_tokens": False,  # We template with special tokens
                                                    "append_concat_token": False, # No need to add additional separator token
                                                    'skip_prepare_dataset': True # skip the dataset preparation
                                                },
                                callbacks=[TrainOnStartCallback()]
                            )
                            # start training, the model will be automatically saved to the hub and the output directory
                            trainer.train()
                            # if push_to_hub == True:
                            #     trainer.push_to_hub()
                            trainer.push_to_hub()
                            # save model
                            # MODEL_DIR = os.getenv('MODEL_DIR', './data/checkpoint')
                            # FINETUNED_MODEL_NAME = os.getenv('FINETUNED_MODEL_NAME',hf_model_id)
                            # chk_path = str(pathlib.Path(MODEL_DIR) / FINETUNED_MODEL_NAME)
                            # print(f"Model is trained and saved as {chk_path}")
                            # trainer.save_model(chk_path)
                            # push to hub
                            
                            # free the memory again
                            del model
                            del trainer
                            torch.cuda.empty_cache()
                            import os, shutil
                            # data_logs = '/app/data/logs'
                            # for filename in os.listdir(data_logs):
                            #     file_path = os.path.join(data_logs, filename)
                            #     try:
                            #         if os.path.isfile(file_path) or os.path.islink(file_path):
                            #             os.unlink(file_path)
                            #         elif os.path.isdir(file_path):
                            #             shutil.rmtree(file_path)
                            #     except Exception as e:
                            #         print('Failed to delete %s. Reason: %s' % (file_path, e))
                            # data_checkpoint = '/app/data/checkpoint'
                            # for filename in os.listdir(data_checkpoint):
                            #     file_path = os.path.join(data_checkpoint, filename)
                            #     try:
                            #         if os.path.isfile(file_path) or os.path.islink(file_path):
                            #             os.unlink(file_path)
                            #         elif os.path.isdir(file_path):
                            #             shutil.rmtree(file_path)
                            #     except Exception as e:
                            #         print('Failed to delete %s. Reason: %s' % (file_path, e))
                            # # Load Model with PEFT adapter
                            # model = AutoPeftModelForCausalLM.from_pretrained(
                            # chk_path,
                            # device_map="auto",
                            # torch_dtype=torch.float16
                            # )
                            # tokenizer = AutoTokenizer.from_pretrained(chk_path)
                            # # load into pipeline
                            # pipe = pipeline(task, model=model, tokenizer=tokenizer)
                            # # Load our test dataset
                            # # eval_dataset = eval_tokenized_datasets #load_dataset("json", data_files="test_dataset.json", split="train")
                            # eval_dataset = train_dataset
                            # rand_idx = randint(0, len(eval_dataset))
                            
                            # # Test on sample
                            # prompt = pipe.tokenizer.apply_chat_template(eval_dataset[rand_idx]["messages"][:2], tokenize=False, add_generation_prompt=True)
                            # outputs = pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
                            
                            # print(f"Query:\n{eval_dataset[rand_idx]['messages'][1]['content']}")
                            # print(f"Original Answer:\n{eval_dataset[rand_idx]['messages'][2]['content']}")
                            # print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")

                            # from tqdm import tqdm
                            
                            
                            # def evaluate(sample):
                            #     prompt = pipe.tokenizer.apply_chat_template(sample["messages"][:2], tokenize=False, add_generation_prompt=True)
                            #     outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
                            #     predicted_answer = outputs[0]['generated_text'][len(prompt):].strip()
                            #     if predicted_answer == sample["messages"][2]["content"]:
                            #         return 1
                            #     else:
                            #         return 0
                            
                            # success_rate = []
                            # number_of_eval_samples = 1000
                            # # iterate over eval dataset and predict
                            # for s in tqdm(eval_dataset.shuffle().select(range(number_of_eval_samples))):
                            #     success_rate.append(evaluate(s))
                            
                            # # compute accuracy
                            # accuracy = sum(success_rate)/len(success_rate)
                            
                            # print(f"Accuracy: {accuracy*100:.2f}%")
                
                train_thread = threading.Thread(target=func_train_model, args=(clone_dir, project_id, token, checkpoint_version, checkpoint_id, dataset_version, dataset_id,model_id,world_size,rank,master_add,master_port,prompt, absolute_path))
                train_thread.start()
                 
                return {"message": "train completed successfully"}
            # except Exception as e:
            #     return {"message": f"train failed: {e}"}
        # elif command.lower() == "stop":
        #     subprocess.run(["pkill", "-9", "-f", "llama_recipes/finetuning.py"])
        #     return {"message": "train stop successfully", "result": "Done"}
        elif command.lower() == "tensorboard":
            def run_tensorboard():
                # train_dir = os.path.join(os.getcwd(), "{project_id}")
                # log_dir = os.path.join(os.getcwd(), "logs")
                p = subprocess.Popen(f"tensorboard --logdir /app/data/logs --host 0.0.0.0 --port=6006", stdout=subprocess.PIPE, stderr=None, shell=True)
                out = p.communicate()
                print(out)
            import threading
            tensorboard_thread = threading.Thread(target=run_tensorboard)
            tensorboard_thread.start()
            return {"message": "tensorboardx started successfully"}
        elif command.lower() == "predict":
            try:
                import torch
            # try:
                # checkpoint = kwargs.get("checkpoint")
                imagebase64 = kwargs.get("image","")
                prompt = kwargs.get("prompt", None)
                model_id = kwargs.get("model_id", "mistralai/Mistral-7B-v0.1")
                text = kwargs.get("text", None)
                token_length = kwargs.get("token_lenght", 30)
                task = kwargs.get("task", "")
                voice = kwargs.get("voice", "")
                max_new_token = kwargs.get("max_new_token", 256)
                temperature = kwargs.get("temperature", 0.2)
                top_k = kwargs.get("top_k", 0.5)
                top_p = kwargs.get("top_p", 0.8)

                if torch.cuda.is_available():
                    if torch.cuda.is_bf16_supported():
                        dtype = torch.bfloat16
                    else:
                        dtype = torch.float16

                    print("CUDA is available.")
                    
                    _model = pipeline(
                        "text-generation",
                        model=model_id, #"meta-llama/Llama-3.2-1B-Instruct", #"meta-llama/Llama-3.2-3B", meta-llama/Llama-3.3-70B-Instruct
                        torch_dtype=dtype, 
                        device_map="auto",  # Hoặc có thể thử "cpu" nếu không ổn,
                        max_new_tokens=int(max_new_token),
                        temperature=float(temperature),
                        top_k=float(top_k),
                        top_p=float(top_p),
                        token='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'
                    )
                else:
                    print("No GPU available, using CPU.")
                    _model = pipeline(
                        "text-generation",
                        model=model_id, #"meta-llama/Llama-3.2-1B-Instruct", #"meta-llama/Llama-3.2-3B", meta-llama/Llama-3.3-70B-Instruct
                        device_map="cpu",
                        max_new_tokens=int(max_new_token),
                        temperature=float(temperature),
                        top_k=float(top_k),
                        top_p=float(top_p),
                        token='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'
                    )

                if len(voice)>0:
                    import base64
                    import requests
                    import torchaudio
                   
                    def decode_base64_to_audio(base64_audio, output_file="output.wav"):
                        # Giải mã Base64 thành nhị phân
                        audio_data = base64.b64decode(base64_audio)
                        
                        # Ghi dữ liệu nhị phân vào file âm thanh
                        with open(output_file, "wb") as audio_file:
                            audio_file.write(audio_data)
                        return output_file
                    
                    audio_file = decode_base64_to_audio(voice["data"])
                    file_path = "unity_on_device.ptl"

                    if not os.path.exists(file_path):
                        url = "https://huggingface.co/facebook/seamless-m4t-unity-small/resolve/main/unity_on_device.ptl"
                        response = requests.get(url)

                        # Lưu file
                        with open("unity_on_device.ptl", "wb") as f:
                            f.write(response.content)

                    audio_input, _ = torchaudio.load(audio_file) # Load waveform using torchaudio

                    s2st_model = torch.jit.load(file_path)

                    with torch.no_grad():
                        prompt, units, waveform = s2st_model(audio_input, tgt_lang="eng")

                predictions = []

                # _model = pipeline("text-generation", model="andrijdavid/Llama3-2B-Base")
                if task == "question-answering":
                    print("Question answering")
                    if text and prompt:
                        generated_text = qa_with_context(_model, text, prompt)
                    elif text and not prompt:
                        generated_text = qa_without_context(_model, text)
                    else:
                        generated_text = qa_with_context(_model, prompt)
                
                elif task == "text-classification":
                    generated_text = text_classification(_model, text, prompt)
                
                elif task == "summarization":
                    generated_text = text_summarization(_model, text)

                else:
                    if not prompt or prompt == "":
                        prompt = text

                    result = _model(prompt, max_length=token_length)
                    generated_text = result[0]['generated_text']

                predictions.append({
                    'result': [{
                        'from_name': "generated_text",
                        'to_name': "text_output",
                        'type': 'textarea',
                        'value': {
                            'text': [generated_text]
                        }
                    }],
                    'model_version': ""
                })

                return {"message": "predict completed successfully", "result": predictions}
            
            except:
                return {"message": "predict failed", "result": None}
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
                return {"message": "prompt_sample completed successfully", "result":prompt_text}
        elif command.lower() == "stop":
            subprocess.run(["pkill", "-9", "-f", "accelerate launch"])
            return {"message": "Done", "result": "Done"}
        elif command.lower() == "action-example":
            return {"message": "Done", "result": "Done"}
        else:
            return {"message": "command not supported", "result": None}
            
            
            # return {"message": "train completed successfully"}
        
    def model(self, project, **kwargs):
        
        import gradio as gr
        from transformers import pipeline
        task = kwargs.get("task", "text-generation")
        model_id = kwargs.get("model_id", "mistralai/Mistral-7B-v0.1")
        project_id = project

        print(f'''\
        Project ID: {project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}''')
        from huggingface_hub import login 
        hf_access_token = kwargs.get("hf_access_token", "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")
        login(token = hf_access_token)
        def load_model(task,model_id, project_id, temperature=None, top_p=None, top_k=None, max_new_token=None):
            from huggingface_hub import login 
            hf_access_token = kwargs.get("hf_access_token", "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")
            login(token = hf_access_token)
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float16

                print("CUDA is available.")
                
                if not temperature:
                    _model = pipeline(
                    task,
                    model=model_id, #"meta-llama/Llama-3.2-1B-Instruct", #"meta-llama/Llama-3.2-3B", meta-llama/Llama-3.3-70B-Instruct
                    torch_dtype=dtype, 
                    device_map="auto",  # Hoặc có thể thử "cpu" nếu không ổn,
                    # max_new_tokens=256,
                    token = "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"
                    )
                else:
                    _model = pipeline(
                        task,
                        model=model_id, #"meta-llama/Llama-3.2-1B-Instruct", #"meta-llama/Llama-3.2-3B", meta-llama/Llama-3.3-70B-Instruct
                        torch_dtype=dtype, 
                        device_map="auto",  # Hoặc có thể thử "cpu" nếu không ổn,
                        # max_new_tokens=256,
                        token = "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",
                        max_new_tokens=int(max_new_token),
                        temperature=float(temperature),
                        top_k=float(top_k),
                        top_p=float(top_p)
                    )
            else:
                print("No GPU available, using CPU.")
                if not temperature:
                    _model = pipeline(
                    task,
                    model=model_id, #"meta-llama/Llama-3.2-1B-Instruct", #"meta-llama/Llama-3.2-3B", meta-llama/Llama-3.3-70B-Instruct
                    torch_dtype=dtype, 
                    device_map="auto",  # Hoặc có thể thử "cpu" nếu không ổn,
                    # max_new_tokens=256,
                    token = "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"
                    )
                else:
                    _model = pipeline(
                        task,
                        model=model_id, #"meta-llama/Llama-3.2-1B-Instruct", #"meta-llama/Llama-3.2-3B", meta-llama/Llama-3.3-70B-Instruct
                        device_map="cpu",
                        # max_new_tokens=256,
                        token = "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",
                        max_new_tokens=int(max_new_token),
                        temperature=float(temperature),
                        top_k=float(top_k),
                        top_p=float(top_p),
                    )
            # try:
            #     channel_deploy = f'project/{project_id}/deploy-history'
            #     client, sub = setup_client(channel_deploy)
            #     send_message(sub,{"refresh": True})
            #     client.disconnect()  # Đóng kết nối client
            # except Exception as e:
            #     print(e)

            return _model
        
        # load_model(task,model_id,project_id)
        
        def generate_response(input_text, temperature, top_p, top_k, max_new_token):
            load_model("text-generation", model_id, project_id, temperature, top_p, top_k, max_new_token)
            # _model = load_model(model_id)
            # prompt_text = f"""
            #        Here is the context: 
            #         {{context}}

            #         Based on the above context, provide an answer to the following question using only a single word or phrase from the context without repeating the question or adding any extra explanation: 
            #         {{question}}

            #         Answer:
            #         """
            # prompt = prompt_text
           
             
            # if not prompt_text or prompt_text == "":
            #     prompt = input_text
            messages = [
                {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
                {"role": "user", "content": input_text},
            ]
            # outputs = _model(
            #     messages,
            #     max_new_tokens=256,
            # )
            result = _model(messages, max_length=1024)
            generated_text = result[0]['generated_text']
             
            # if input_text and prompt_text:
            #     generated_text = qa_with_context(_model, input_text, prompt_text)
            # elif input_text and not prompt_text:
            #     generated_text = qa_without_context(_model, prompt_text)
            # else:
            #     generated_text = qa_with_context(_model, prompt_text)

            return generated_text
        
        def summarization_response(input_text, temperature, top_p, top_k, max_new_token):
            load_model("text-generation", model_id, project_id, temperature, top_p, top_k, max_new_token)
            # _model = load_model(model_id)
            generated_text = text_summarization(_model, input_text)
  
            return generated_text
        
        def text_classification_response(input_text,categories_text, temperature, top_p, top_k, max_new_token):
            load_model("text-generation", model_id, project_id, temperature, top_p, top_k, max_new_token)  
            # prompt_text = f"""
            #         Summarize the following text into a single, concise paragraph focusing on the key ideas and important points:

            #         Text: 
            #         {{context}}

            #         Summary:
            #         """
                
            # _model = load_model(model_id)
            generated_text = text_classification(_model, input_text, categories_text)
            return generated_text
        
        def question_answering_response(context_textbox,question_textbox, temperature, top_p, top_k, max_new_token):
            load_model("text-generation", model_id, project_id, temperature, top_p, top_k, max_new_token)
            # _model = load_model(model_id)
            if input_text and question_textbox:
                generated_text = qa_with_context(_model, context_textbox, question_textbox)
            elif context_textbox and not question_textbox:
                generated_text = qa_without_context(_model, question_textbox)
            else:
                generated_text = qa_with_context(_model, question_textbox)

            return generated_text
        
        with gr.Blocks() as demo_text_generation:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                        # prompt_text = gr.Textbox(label="Prompt text")
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=100.0,
                            step=0.1,
                            value=0.9
                        )
                        top_p = gr.Slider(
                            label="Top_p",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.6
                        )
                        top_k = gr.Slider(
                            label="Top_k",
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=0
                        )
                        max_new_token = gr.Slider(
                            label="Max new tokens",
                            minimum=1,
                            maximum=1024,
                            step=1,
                            value=256
                        )
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Output text")
            # gr.on(
            #     triggers=[input_text.submit, temperature.submit, top_p.submit, top_k.submit, max_new_token.submit, btn.click],
            #     fn=generate_response,
            #     inputs=[input_text, temperature, top_p, top_k, max_new_token],
            #     outputs=output_text,
            #     api_name=task,
            # )
            btn.click(
                fn=generate_response,
                inputs=[input_text, temperature, top_p, top_k, max_new_token],
                outputs=output_text
            )
        
        with gr.Blocks() as demo_summarization:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=100.0,
                            step=0.1,
                            value=0.9
                        )
                        top_p = gr.Slider(
                            label="Top_p",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.6
                        )
                        top_k = gr.Slider(
                            label="Top_k",
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=0
                        )
                        max_new_token = gr.Slider(
                            label="Max new tokens",
                            minimum=1,
                            maximum=1024,
                            step=1,
                            value=256
                        )
                        # prompt_text = gr.Textbox(label="Prompt text")
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Output text")
            # gr.on(
            #     triggers=[input_text.submit, temperature.submit, top_p.submit, top_k.submit, max_new_token.submit, btn.click],
            #     fn=summarization_response,
            #     inputs=[input_text, temperature, top_p, top_k, max_new_token],
            #     outputs=output_text,
            #     api_name=task,
            # )
            btn.click(
                fn=summarization_response,
                inputs=[input_text, temperature, top_p, top_k, max_new_token],
                outputs=output_text
            )
        
        with gr.Blocks() as demo_question_answering:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        context_textbox = gr.Textbox(label="Context text")
                        question_textbox = gr.Textbox(label="Question text")
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=100.0,
                            step=0.1,
                            value=0.9
                        )
                        top_p = gr.Slider(
                            label="Top_p",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.6
                        )
                        top_k = gr.Slider(
                            label="Top_k",
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=0
                        )
                        max_new_token = gr.Slider(
                            label="Max new tokens",
                            minimum=1,
                            maximum=1024,
                            step=1,
                            value=256
                        )
                       
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text =   gr.Textbox(label="Response:")
            

            # gr.Examples(
            #    inputs=[input_text],
            #     outputs=output_text,
            #     fn=question_answering_response,
            #     api_name=False,
            # )

            # gr.on(
            #     triggers=[context_textbox.submit,question_textbox.submit, temperature.submit, top_p.submit, top_k.submit, max_new_token.submit, btn.click],
            #     fn=question_answering_response,
            #     inputs=[context_textbox, question_textbox, temperature, top_p, top_k, max_new_token],
            #     outputs=output_text,
            #     api_name=task,
            # )
            btn.click(
                fn=question_answering_response,
                inputs=[context_textbox, question_textbox, temperature, top_p, top_k, max_new_token],
                outputs=output_text
            )
        
        with gr.Blocks() as demo_text_classification:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                        categories_text = gr.Textbox(label="Categories text")
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=100.0,
                            step=0.1,
                            value=0.9
                        )
                        top_p = gr.Slider(
                            label="Top_p",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.6
                        )
                        top_k = gr.Slider(
                            label="Top_k",
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=0
                        )
                        max_new_token = gr.Slider(
                            label="Max new tokens",
                            minimum=1,
                            maximum=1024,
                            step=1,
                            value=256
                        )
                # with gr.Column():
                #     with gr.Group():
                #         input_text = gr.Textbox(label="Input text")
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Response:")

            
            # gr.on(
            #     triggers=[input_text.submit,categories_text.submit, temperature.submit, top_p.submit, top_k.submit, max_new_token.submit, btn.click],
            #     fn=text_classification_response,
            #     inputs=[input_text, categories_text, temperature, top_p, top_k, max_new_token],
            #     outputs=output_text,
            #     api_name=task,
            # )
            btn.click(
                fn=text_classification_response,
                inputs=[input_text, categories_text, temperature, top_p, top_k, max_new_token],
                outputs=output_text
            )
        
        def sentiment_classifier(text, temperature, top_p, top_k, max_new_token):
            try:
                sentiment_classifier = pipeline("sentiment-analysis", temperature=temperature, top_p=top_p, top_k=top_k, max_new_token=max_new_token)
                sentiment_response = sentiment_classifier(text)
                # label = sentiment_response[0]['label']
                # score = sentiment_response[0]['score']
                print(sentiment_response)
                import json
                return json.dumps(sentiment_response)
            except Exception as e:
                return str(e)
        
        with gr.Blocks() as demo_sentiment_analysis:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=100.0,
                            step=0.1,
                            value=0.9
                        )
                        top_p = gr.Slider(
                            label="Top_p",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.6
                        )
                        top_k = gr.Slider(
                            label="Top_k",
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=0
                        )
                        max_new_token = gr.Slider(
                            label="Max new tokens",
                            minimum=1,
                            maximum=1024,
                            step=1,
                            value=256
                        )
                    btn = gr.Button("Submit")
                with gr.Column():
                    
                    label_text = gr.Label(label="Label: ")
                    score_text = gr.Label(label="Score: ")

            # gr.Examples(
            #     inputs=[input_text, source_language, target_language],
            #     outputs=output_text,
            #     fn=generate_response,
            #     api_name=False,
            # )

            # gr.on(
            #     triggers=[input_text.submit, temperature.submit, top_p.submit, top_k.submit, max_new_token.submit, btn.click],
            #     fn=sentiment_classifier,
            #     inputs=[input_text, temperature, top_p, top_k, max_new_token],
            #     outputs=[label_text, score_text],
            #     api_name=task,
            # )
            btn.click(
                fn=sentiment_classifier,
                inputs=[input_text, temperature, top_p, top_k, max_new_token],
                outputs=output_text
            )

        def predict_entities(input_text, categories_text, temperature, top_p, top_k, max_new_token):
            #  # Initialize the text-generation pipeline with your model
            # pipe = pipeline(task, model=model_id)
            # # Use the loaded model to identify entities in the text
            # entities = pipe(text)
            # # Highlight identified entities in the input text
            # highlighted_text = text
            # for entity in entities:
            #     entity_text = text[entity['start']:entity['end']]
            #     replacement = f"<span style='border: 2px solid green;'>{entity_text}</span>"
            #     highlighted_text = highlighted_text.replace(entity_text, replacement)
            # return highlighted_text
            load_model("text-generation", model_id, project_id, temperature, top_p, top_k, max_new_token)
            generated_text = text_ner(_model, input_text, categories_text)
            return generated_text
        
        with gr.Blocks() as demo_ner:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                        categories_text = gr.Textbox(label="Categories text")
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=100.0,
                            step=0.1,
                            value=0.9
                        )
                        top_p = gr.Slider(
                            label="Top_p",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.6
                        )
                        top_k = gr.Slider(
                            label="Top_k",
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=0
                        )
                        max_new_token = gr.Slider(
                            label="Max new tokens",
                            minimum=1,
                            maximum=1024,
                            step=1,
                            value=256
                        )
                    # with gr.Group():
                    #     input_text = gr.Textbox(label="Input text")
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Response:")

            # gr.Examples(
            #     inputs=[input_text],
            #     outputs=output_text,
            #     fn=generate_response,
            #     api_name=False,
            # )

            # gr.on(
            #     triggers=[input_text.submit,categories_text.submit, temperature.submit, top_p.submit, top_k.submit, max_new_token.submit, btn.click],
            #     fn=predict_entities,
            #     inputs=[input_text, categories_text, temperature, top_p, top_k, max_new_token],
            #     outputs=output_text,
            #     api_name=task,
            # )
            btn.click(
                fn=predict_entities,
                inputs=[input_text, categories_text, temperature, top_p, top_k, max_new_token],
                outputs=output_text
            )
        
        with gr.Blocks() as demo_text2text_generation:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text", placeholder="Enter your text here")
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=100.0,
                            step=0.1,
                            value=0.9
                        )
                        top_p = gr.Slider(
                            label="Top_p",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.6
                        )
                        top_k = gr.Slider(
                            label="Top_k",
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=0
                        )
                        max_new_token = gr.Slider(
                            label="Max new tokens",
                            minimum=1,
                            maximum=1024,
                            step=1,
                            value=256
                        )
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Output text")
            
            # Gắn sự kiện với nút Submit
            btn.click(
                fn=generate_response,
                inputs=[input_text, temperature, top_p, top_k, max_new_token],
                outputs=output_text
            )

        def chatbot_continuous_chat(history, user_input, temperature, top_p, top_k, max_new_token):
            load_model("text-generation", model_id, project_id, temperature, top_p, top_k, max_new_token)
            generated_text = chatbot_with_history(_model, history, user_input)
            return generated_text
        
        with gr.Blocks() as demo_chatbot:
            chatbot = gr.Chatbot(type="messages")
            with gr.Row():
                user_input = gr.Textbox(label="Your message", placeholder="Type your message here...")
            with gr.Row():
                temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=100.0,
                            step=0.1,
                            value=0.9
                        )
                top_p = gr.Slider(
                    label="Top_p",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.6
                )
                top_k = gr.Slider(
                    label="Top_k",
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=0
                )
                max_new_token = gr.Slider(
                    label="Max new tokens",
                    minimum=1,
                    maximum=1024,
                    step=1,
                    value=256
                )
            with gr.Row():
                btn = gr.Button("Send")

            # Bind function to button click
            btn.click(
                fn=chatbot_continuous_chat,
                inputs=[chatbot, user_input, temperature, top_p, top_k, max_new_token],  # Thêm các tham số vào hàm
                outputs=[chatbot, user_input]  # Cập nhật lại chatbot với phản hồi mới và xóa trường nhập liệu
            )

        DESCRIPTION = """\
        # LLM UI
        This is a demo of LLM UI.
        """
        with gr.Blocks(css="style.css") as demo:
            gr.Markdown(DESCRIPTION)

            with gr.Tabs():
                if task == "text-generation":
                  with gr.Tab(label=task):
                        demo_text_generation.render()
                elif task == "summarization":
                  with gr.Tab(label=task):
                        demo_summarization.render()
                elif task == "question-answering":
                   with gr.Tab(label=task):
                        demo_question_answering.render()
                elif task == "text-classification":
                    with gr.Tab(label=task):
                            demo_text_classification.render()
                elif task == "sentiment-analysis":
                  with gr.Tab(label=task):
                        demo_sentiment_analysis.render()
                elif task == "ner":
                   with gr.Tab(label=task):
                        demo_ner.render()
                # elif task == "fill-mask":
                #   with gr.Tab(label=task):
                #         demo_fill_mask.render()
                elif task == "text2text-generation":
                   with gr.Tab(label=task):
                        demo_text2text_generation.render()
                else:
                    return {"share_url": "", 'local_url': ""}
                
        gradio_app, local_url, share_url = demo.launch(share=True, quiet=True, prevent_thread_lock=True, server_name='0.0.0.0',show_error=True)
   
        return {"share_url": share_url, 'local_url': local_url}
    
    def model_trial(self, project, **kwargs):
        import gradio as gr 

        return {"message": "Done", "result": "Done"}


        css = """
        .feedback .tab-nav {
            justify-content: center;
        }

        .feedback button.selected{
            background-color:rgb(115,0,254); !important;
            color: #ffff !important;
        }

        .feedback button{
            font-size: 16px !important;
            color: black !important;
            border-radius: 12px !important;
            display: block !important;
            margin-right: 17px !important;
            border: 1px solid var(--border-color-primary);
        }

        .feedback div {
            border: none !important;
            justify-content: center;
            margin-bottom: 5px;
        }

        .feedback .panel{
            background: none !important;
        }


        .feedback .unpadded_box{
            border-style: groove !important;
            width: 500px;
            height: 345px;
            margin: auto;
        }

        .feedback .secondary{
            background: rgb(225,0,170);
            color: #ffff !important;
        }

        .feedback .primary{
            background: rgb(115,0,254);
            color: #ffff !important;
        }

        .upload_image button{
            border: 1px var(--border-color-primary) !important;
        }
        .upload_image {
            align-items: center !important;
            justify-content: center !important;
            border-style: dashed !important;
            width: 500px;
            height: 345px;
            padding: 10px 10px 10px 10px
        }
        .upload_image .wrap{
            align-items: center !important;
            justify-content: center !important;
            border-style: dashed !important;
            width: 500px;
            height: 345px;
            padding: 10px 10px 10px 10px
        }

        .webcam_style .wrap{
            border: none !important;
            align-items: center !important;
            justify-content: center !important;
            height: 345px;
        }

        .webcam_style .feedback button{
            border: none !important;
            height: 345px;
        }

        .webcam_style .unpadded_box {
            all: unset !important;
        }

        .btn-custom {
            background: rgb(0,0,0) !important;
            color: #ffff !important;
            width: 200px;
        }

        .title1 {
            margin-right: 90px !important;
        }

        .title1 block{
            margin-right: 90px !important;
        }

        """

        with gr.Blocks(css=css) as demo:
            with gr.Row():
                with gr.Column(scale=10):
                    gr.Markdown(
                        """
                        # Theme preview: `AIxBlock`
                        """
                    )

            import numpy as np
            def predict(input_img):
                import cv2
                result = self.action(project, "predict",collection="",data={"img":input_img})
                print(result)
                if result['result']:
                    boxes = result['result']['boxes']
                    names = result['result']['names']
                    labels = result['result']['labels']
                    
                    for box, label in zip(boxes, labels):
                        box = [int(i) for i in box]
                        label = int(label)
                        input_img = cv2.rectangle(input_img, box, color=(255, 0, 0), thickness=2)
                        # input_img = cv2.(input_img, names[label], (box[0], box[1]), color=(255, 0, 0), size=1)
                        input_img = cv2.putText(input_img, names[label], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                return input_img
            
            def download_btn(evt: gr.SelectData):
                print(f"Downloading {dataset_choosen}")
                return f'<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"><a href="/my_ml_backend/datasets/{evt.value}" style="font-size:50px"> <i class="fa fa-download"></i> Download this dataset</a>'
                
            def trial_training(dataset_choosen):
                print(f"Training with {dataset_choosen}")
                result = self.action(project, "train",collection="",data=dataset_choosen)
                return result['message']

            def get_checkpoint_list(project):
                print("GETTING CHECKPOINT LIST")
                print(f"Proejct: {project}")
                import os
                checkpoint_list = [i for i in os.listdir("my_ml_backend/models") if i.endswith(".pt")]
                checkpoint_list = [f"<a href='./my_ml_backend/checkpoints/{i}' download>{i}</a>" for i in checkpoint_list]
                if os.path.exists(f"my_ml_backend/{project}"):
                    for folder in os.listdir(f"my_ml_backend/{project}"):
                        if "train" in folder:
                            project_checkpoint_list = [i for i in os.listdir(f"my_ml_backend/{project}/{folder}/weights") if i.endswith(".pt")]
                            project_checkpoint_list = [f"<a href='./my_ml_backend/{project}/{folder}/weights/{i}' download>{folder}-{i}</a>" for i in project_checkpoint_list]
                            checkpoint_list.extend(project_checkpoint_list)
                
                return "<br>".join(checkpoint_list)

            def tab_changed(tab):
                if tab == "Download":
                    get_checkpoint_list(project=project)
            
            def upload_file(file):
                return "File uploaded!"
            
            with gr.Tabs(elem_classes=["feedback"]) as parent_tabs:
                with gr.TabItem("Image", id=0):   
                    with gr.Row():
                        gr.Markdown("## Input", elem_classes=["title1"])
                        gr.Markdown("## Output", elem_classes=["title1"])
                    
                    gr.Interface(predict, gr.Image(elem_classes=["upload_image"], sources="upload", container = False, height = 345,show_label = False), 
                                gr.Image(elem_classes=["upload_image"],container = False, height = 345,show_label = False), allow_flagging = False             
                    )


                # with gr.TabItem("Webcam", id=1):    
                #     gr.Image(elem_classes=["webcam_style"], sources="webcam", container = False, show_label = False, height = 450)

                # with gr.TabItem("Video", id=2):    
                #     gr.Image(elem_classes=["upload_image"], sources="clipboard", height = 345,container = False, show_label = False)

                # with gr.TabItem("About", id=3):  
                #     gr.Label("About Page")

                with gr.TabItem("Trial Train", id=2):
                    gr.Markdown("# Trial Train")
                    with gr.Column():
                        with gr.Column():
                            gr.Markdown("## Dataset template to prepare your own and initiate training")
                            with gr.Row():
                                #get all filename in datasets folder
                                if not os.path.exists(f"./datasets"):
                                    os.makedirs(f"./datasets")
                                datasets = [(f"dataset{i}", name) for i, name in enumerate(os.listdir('./datasets'))]
                                
                                dataset_choosen = gr.Dropdown(datasets, label="Choose dataset", show_label=False, interactive=True, type="value")
                                # gr.Button("Download this dataset", variant="primary").click(download_btn, dataset_choosen, gr.HTML())
                                download_link = gr.HTML("""
                                        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
                                        <a href='' style="font-size:24px"><i class="fa fa-download" ></i> Download this dataset</a>""")
                                
                                dataset_choosen.select(download_btn, None, download_link)
                                
                                #when the button is clicked, download the dataset from dropdown
                                # download_btn
                            gr.Markdown("## Upload your sample dataset to have a trial training")
                            # gr.File(file_types=['tar','zip'])
                            gr.Interface(predict, gr.File(elem_classes=["upload_image"],file_types=['tar','zip']), 
                                gr.Label(elem_classes=["upload_image"],container = False), allow_flagging = False             
                    )
                            with gr.Row():
                                gr.Markdown(f"## You can attemp up to {2} FLOps")
                                gr.Button("Trial Train", variant="primary").click(trial_training, dataset_choosen, None)
                
                # with gr.TabItem("Download"):
                #     with gr.Column():
                #         gr.Markdown("## Download")
                #         with gr.Column():
                #             gr.HTML(get_checkpoint_list(project))

        gradio_app, local_url, share_url = demo.launch(share=True, quiet=True, prevent_thread_lock=True, server_name='0.0.0.0',show_error=True)
   
        return {"share_url": share_url, 'local_url': local_url}
    def download(self, project, **kwargs):
        from flask import send_from_directory,request
        file_path = request.args.get('path')
        print(request.args)
        return send_from_directory(os.getcwd(), file_path)
    # def download(self, project, **kwargs):
    #     from flask import send_file
    #     file_path = kwargs.get("path","")
    #     if os.path.exists(os.path.join("/app/", file_path)):
    #         return send_file(os.path.join("/app/", file_path), as_attachment=True)
    #     else:
    #         print("file dont exist")
    #         return {"msg": "file dont exist"}