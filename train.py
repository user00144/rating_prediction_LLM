import torch
from transformers import get_linear_schedule_with_warmup
from dataset import get_dataloader
from model import get_model
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings(action='ignore')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = "cuda"
max_length = 1024
lr = 1e-3
num_epochs = 3
batch_size = 4
data_path = "./dataset/Amazon_movies_and_tv/Movies_and_TV.jsonl"

model, tokenizer = get_model("declare-lab/flan-alpaca-gpt4-xl")
train_dataloader, eval_dataloader, test_dataloader = get_dataloader(data_path, tokenizer, max_length, batch_size, p_val = 0.3)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

# training and evaluation
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")


torch.save(model.state_dict(), 'model_01.pt')


print("=================== Test ===================")

import re
import numpy as np

#train_complete
model = model.merge_and_unload()

record_rate = 100

gen_out = []

test_total_loss = 0

def loss_RMSE(actual, predicted) :
    return np.sqrt(np.square(actual-predicted))

for step, batch in enumerate(tqdm(test_dataloader)) :
    output = model.to(device).generate(batch["input_ids"].squeeze(0).to(device))
    output_decode = tokenizer.decode(output.squeeze(0))
    f_out = np.array(list(map(float,re.findall(r"[0-9.]+",output_decode))))
    pred = 0
    if f_out.any() :
        pred = np.sum(f_out)
        pred = pred / len(f_out)
    else :
        pred = 0
    if step % record_rate == 0 :
        gen_out.append(f"in {step} answer : {output_decode} / f_out : {f_out} / mean : {pred}")
    actual = batch["label"].detach().cpu().numpy()
    loss = loss_RMSE(actual, pred)
    test_total_loss += loss

with open("./gen_out.txt", 'w+') as file:
    file.write('\n'.join(gen_out)) 

test_loss = test_total_loss / len(test_dataloader)
print(f"test_rating_RMSE : {test_loss}")