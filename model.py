from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

def get_model(model_id) :
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer