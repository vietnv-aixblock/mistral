import random
from typing import List, Dict, Optional
from aixblock_ml.model import AIxBlockMLBase
# from ultralytics import YOLO
from transformers import pipeline,BloomForCausalLM,BloomTokenizerFast
import os
 
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt import qa_with_context, text_classification, text_summarization, qa_without_context
# device = "cuda" # the device to load the model onto

# model = AutoModelForCausalLM.from_pretrained("djuna/Qwen2-2B-Instruct", device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained("djuna/Qwen2-2B-Instruct")

# prompt = "Give me a short introduction to large language model."

# messages = [{"role": "user", "content": prompt}]

# text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# model_inputs = tokenizer([text], return_tensors="pt").to(device)

# generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True)

# generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

from huggingface_hub import HfFolder
import torch

# Đặt token của bạn vào đây
hf_token = os.getenv("HF_TOKEN", "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")
# Lưu token vào local
HfFolder.save_token(hf_token)
print("Login successful")

if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    print("CUDA is available.")
    _model = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-v0.1",
        torch_dtype=dtype, 
        device_map="auto",
        max_new_tokens=50
    )
else:
    print("No GPU available, using CPU.")
    _model = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-v0.1",
        device_map="cpu",
        max_new_tokens=50
    )

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
#             {
#     "command": "train",
#     "params": {
#         "model_id":"appvoid/llama-3-1b",
#         "dataset_id":"lucasmccabe-lmi/CodeAlpaca-20k",
#         "task": "text-generation",
#         "num_train_epochs":3,
#         "per_device_train_batch_size":3,
#         "gradient_accumulation_steps":2,
#         "gradient_checkpointing":true,
#         "optim":"adamw_torch_fused",
#         "logging_steps":10,
#         "save_strategy":"epoch",
#         "learning_rate":2e-4,
#         "bf16":false,
#         "fp16":false, 
#         "max_grad_norm":0.3,
#         "warmup_ratio":0.03,    
#         "lora_alpha":128,
#         "lora_dropout":0.05,
#         "bias":"none",
#         "target_modules":"all-linear",
#         "task_type":"CAUSAL_LM",
#         "prompt":"",
#         "customize_fields":"",
        # "hf_model_id":"tonyshark/llama3"
#     },
#     "project": "237"
# }
                model_id = kwargs.get("model_id", "mistralai/Mistral-7B-v0.1")  #"tiiuae/falcon-7b" "bigscience/bloomz-1b7" `zanchat/falcon-1b` `appvoid/llama-3-1b` meta-llama/Llama-3.2-3B` `mistralai/Mistral-7B-v0.1` `bigscience/bloomz-1b7` `Qwen/Qwen2-1.5B`
                dataset_id = kwargs.get("dataset_id","lucasmccabe-lmi/CodeAlpaca-20k") #gingdev/llama_vi_52k kigner/ruozhiba-llama3-tt
                num_train_epochs = kwargs.get("num_train_epochs", 3)
                per_device_train_batch_size = kwargs.get("per_device_train_batch_size", 3)
                gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 2)
                gradient_checkpointing = kwargs.get("gradient_checkpointing", True)
                optim = kwargs.get("optim", "adamw_torch_fused")
                logging_steps = kwargs.get("logging_steps", 10)
                save_strategy = kwargs.get("save_strategy", "epoch")
                learning_rate = kwargs.get("learning_rate", 2e-4)
                bf16 = kwargs.get("bf16", False)
                fp16 = kwargs.get("fp16", False)
                max_grad_norm = kwargs.get("max_grad_norm", 0.3)
                warmup_ratio = kwargs.get("warmup_ratio", 0.03)
                lora_alpha = kwargs.get("lora_alpha", 128)
                lora_dropout = kwargs.get("lora_dropout", 0.05)
                bias = kwargs.get("bias", "none")
                target_modules = kwargs.get("target_modules", "all-linear")
                task_type = kwargs.get("task_type", "CAUSAL_LM")
                use_cpu = kwargs.get("use_cpu", True)
                push_to_hub = kwargs.get("push_to_hub", False)
                hf_model_id = kwargs.get("hf_model_id", "tonyshark/llama3")
                
                task = kwargs.get("task", "text-generation")
                remove_unused_columns =kwargs.get("remove_unused_columns", False)
                max_seq_length = kwargs.get("max_seq_length", 1024)
               
            # try:
                import threading
                clone_dir = os.path.join(os.getcwd())
                # epochs = kwargs.get("num_epochs", 10)
                project_id = kwargs.get("project_id")
                token = kwargs.get("token")
                checkpoint_version = kwargs.get("checkpoint_version")
                checkpoint_id = kwargs.get("checkpoint")
                dataset_version = kwargs.get("dataset_version")
                # dataset_id = kwargs.get("dataset")
                channel_log = kwargs.get("channel_log", "training_logs")
                world_size = kwargs.get("world_size", "1")
                rank = kwargs.get("rank", "0")
                master_add = kwargs.get("master_add")
                master_port = kwargs.get("master_port", "12345")
                # entry_file = kwargs.get("entry_file")
                configs = kwargs.get("configs")
                def func_train_model(clone_dir, project_id, epochs, token, checkpoint_version, checkpoint_id, dataset_version, dataset_id,model_id,learning_rate,per_device_train_batch_size,use_cpu,bf16,fp16,push_to_hub,task,hf_model_id,num_train_epochs):
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
                        model_id = "appvoid/llama-3-1b" # or  `appvoid/llama-3-1b` tiiuae/falcon-7b` `mistralai/Mistral-7B-v0.1` `bigscience/bloomz-1b7` `Qwen/Qwen2-1.5B`
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
                
                    print("Data is formatted and ready!")
                    
                    def compute_metrics(eval_pred):
                        predictions, labels = eval_pred
                        predictions = np.argmax(predictions, axis=1)
                        return accuracy.compute(predictions=predictions, references=labels)
                    
                    class TrainOnStartCallback(TrainerCallback):
                        def on_train_begin(self, args, state, control, logs=None, **kwargs):
                            # Log training loss at step 0
                            logs = logs or {}
                            self.log(logs)

                        def log(self, logs):
                            print(f"Logging at start: {logs}")
                    training_args = TrainingArguments(
                        output_dir= f"./data/checkpoint/{model_id}", # directory to save and repository id
                        learning_rate=learning_rate,
                        per_device_train_batch_size=per_device_train_batch_size,
                        per_device_eval_batch_size=16,
                        num_train_epochs=num_train_epochs,
                        weight_decay=0.01,
                        save_strategy="epoch", 
                        report_to="tensorboard",
                        use_cpu=use_cpu,
                        bf16 = bf16,
                        fp16 = fp16
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
                        compute_metrics=compute_metrics,
                        dataset_kwargs={
                                            "add_special_tokens": False,  # We template with special tokens
                                            "append_concat_token": False, # No need to add additional separator token
                                            'skip_prepare_dataset': True # skip the dataset preparation
                                        },
                        callbacks=[TrainOnStartCallback()]
                    )
                    # start training, the model will be automatically saved to the hub and the output directory
                    trainer.train()
                    
                    # save model
                    MODEL_DIR = os.getenv('MODEL_DIR', './data/checkpoint')
                    FINETUNED_MODEL_NAME = os.getenv('FINETUNED_MODEL_NAME',hf_model_id)
                    chk_path = str(pathlib.Path(MODEL_DIR) / FINETUNED_MODEL_NAME)
                    print(f"Model is trained and saved as {chk_path}")
                    trainer.save_model(chk_path)
                    # push to hub
                    if push_to_hub == True:
                        trainer.push_to_hub()
                    # free the memory again
                    del model
                    del trainer
                    torch.cuda.empty_cache()
                    # Load Model with PEFT adapter
                    model = AutoPeftModelForCausalLM.from_pretrained(
                    chk_path,
                    device_map="auto",
                    torch_dtype=torch.float16
                    )
                    tokenizer = AutoTokenizer.from_pretrained(chk_path)
                    # load into pipeline
                    pipe = pipeline(task, model=model, tokenizer=tokenizer)
                    # Load our test dataset
                    # eval_dataset = eval_tokenized_datasets #load_dataset("json", data_files="test_dataset.json", split="train")
                    eval_dataset = train_dataset
                    rand_idx = randint(0, len(eval_dataset))
                    
                    # Test on sample
                    prompt = pipe.tokenizer.apply_chat_template(eval_dataset[rand_idx]["messages"][:2], tokenize=False, add_generation_prompt=True)
                    outputs = pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
                    
                    print(f"Query:\n{eval_dataset[rand_idx]['messages'][1]['content']}")
                    print(f"Original Answer:\n{eval_dataset[rand_idx]['messages'][2]['content']}")
                    print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")

                    from tqdm import tqdm
                    
                    
                    def evaluate(sample):
                        prompt = pipe.tokenizer.apply_chat_template(sample["messages"][:2], tokenize=False, add_generation_prompt=True)
                        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
                        predicted_answer = outputs[0]['generated_text'][len(prompt):].strip()
                        if predicted_answer == sample["messages"][2]["content"]:
                            return 1
                        else:
                            return 0
                    
                    success_rate = []
                    number_of_eval_samples = 1000
                    # iterate over eval dataset and predict
                    for s in tqdm(eval_dataset.shuffle().select(range(number_of_eval_samples))):
                        success_rate.append(evaluate(s))
                    
                    # compute accuracy
                    accuracy = sum(success_rate)/len(success_rate)
                    
                    print(f"Accuracy: {accuracy*100:.2f}%")

                train_thread = threading.Thread(target=func_train_model, args=(clone_dir, project_id, num_train_epochs, token, checkpoint_version, checkpoint_id, dataset_version, dataset_id,model_id,learning_rate,per_device_train_batch_size,use_cpu,bf16,fp16,push_to_hub,task,hf_model_id,num_train_epochs ))
                train_thread.start()

                return {"message": "train completed successfully"}
          
        elif command.lower() == "predict":
            try:
                import torch
            # try:
                # checkpoint = kwargs.get("checkpoint")
                imagebase64 = kwargs.get("image","")
                prompt = kwargs.get("prompt", "")
                model_id = kwargs.get("model_id", "")
                text = kwargs.get("text", "")
                token_length = kwargs.get("token_lenght", 30)
                task = kwargs.get("task", "")
                voice = kwargs.get("voice", "")


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

                    Based on the above context, provide an answer to the following question using only a single word or phrase from the context without repeating the question or adding any extra explanation: 
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
        else:
            return {"message": "command not supported", "result": None}
            
            
            # return {"message": "train completed successfully"}
        
    def model(self, project, **kwargs):
        
        import gradio as gr
        from transformers import pipeline
        task = kwargs.get("task", "")
        model_id = kwargs.get("model_id", "")
        from huggingface_hub import login 
        hf_access_token = kwargs.get("hf_access_token", "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")
        login(token = hf_access_token)
        def generate_response(input_text,prompt_text):
            prompt = prompt_text
            if not prompt_text or prompt_text == "":
                prompt = input_text

            result = _model(prompt, max_length=100)
            generated_text = result[0]['generated_text']
             
            if input_text and prompt_text:
                generated_text = qa_with_context(_model, input_text, prompt_text)
            elif input_text and not prompt_text:
                generated_text = qa_without_context(_model, prompt_text)
            else:
                generated_text = qa_with_context(_model, prompt_text)

            return generated_text
        def summarization_response(input_text,prompt_text):
            generated_text = text_summarization(_model, input_text)
            import json
            return json.dumps(generated_text)
        def text_classification_response(input_text,prompt_text):

            generated_text = text_classification(_model, input_text, prompt_text)
            import json
            return json.dumps(generated_text)
        def question_answering_response(context_textbox,question_textbox):
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
                        prompt_text = gr.Textbox(label="Prompt text")
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Output text")
            gr.on(
                triggers=[input_text.submit,prompt_text.submit, btn.click],
                fn=generate_response,
                inputs=[input_text,prompt_text],
                outputs=output_text,
                api_name=task,
            )
        with gr.Blocks() as demo_summarization:
            with gr.Row():
                
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                        prompt_text = gr.Textbox(label="Prompt text")
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Output text")
            gr.on(
                triggers=[input_text.submit,prompt_text.submit, btn.click],
                fn=summarization_response,
                inputs=[input_text,prompt_text],
                outputs=output_text,
                api_name=task,
            )
        with gr.Blocks() as demo_question_answering:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        context_textbox = gr.Textbox(label="Context text")
                        question_textbox = gr.Textbox(label="Question text")
                       
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text =   gr.Textbox(label="Response:")
            

            # gr.Examples(
            #    inputs=[input_text],
            #     outputs=output_text,
            #     fn=question_answering_response,
            #     api_name=False,
            # )

            gr.on(
                triggers=[input_text.submit, btn.click],
                fn=question_answering_response,
                inputs=[context_textbox,question_textbox],
                outputs=output_text,
                api_name=task,
            )
        
        with gr.Blocks() as demo_text_classification:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Response:")

            
            gr.on(
                triggers=[input_text.submit, btn.click],
                fn=text_classification_response,
                inputs=[input_text],
                outputs=output_text,
                api_name=task,
            )
        def sentiment_classifier(text):
            try:
                sentiment_classifier = pipeline("sentiment-analysis")
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

            gr.on(
                triggers=[input_text.submit, btn.click],
                fn=sentiment_classifier,
                inputs=[input_text],
                outputs=[label_text,score_text],
                api_name=task,
            )
        def predict_entities(text):
             # Initialize the text-generation pipeline with your model
            pipe = pipeline(task, model=model_id)
            # Use the loaded model to identify entities in the text
            entities = pipe(text)
            # Highlight identified entities in the input text
            highlighted_text = text
            for entity in entities:
                entity_text = text[entity['start']:entity['end']]
                replacement = f"<span style='border: 2px solid green;'>{entity_text}</span>"
                highlighted_text = highlighted_text.replace(entity_text, replacement)
            return highlighted_text
        with gr.Blocks() as demo_ner:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.HTML()

            # gr.Examples(
            #     inputs=[input_text],
            #     outputs=output_text,
            #     fn=generate_response,
            #     api_name=False,
            # )

            gr.on(
                triggers=[input_text.submit, btn.click],
                fn=predict_entities,
                inputs=[input_text],
                outputs=output_text,
                api_name=task,
            )
        
        with gr.Blocks() as demo_text2text_generation:
            with gr.Row():
                
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                        prompt_text = gr.Textbox(label="Prompt text")
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Output text")
            gr.on(
                triggers=[input_text.submit,prompt_text.submit, btn.click],
                fn=generate_response,
                inputs=[input_text,prompt_text],
                outputs=output_text,
                api_name=task,
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
                                datasets = [(f"dataset{i}", name) for i, name in enumerate(os.listdir('./my_ml_backend/datasets'))]
                                
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
        return super().download(project, **kwargs)