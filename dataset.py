from torch.utils.data import Dataset, DataLoader
import jsonlines
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class CustomDataset(Dataset) :
    def __init__(self, data, tokenizer, max_length) :
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def make_prompt(self, review) :
        user_id = review['user_id']
        movie_title = review['title']
        text = review['text']
        prompt = f"User id : {user_id}\n Target : {movie_title}\n Review text : {text}\n Prompt : According to above information, what star rating would the user {user_id} give? "
        return prompt
        
    def tokenize(self, prompt, label) :
        model_inputs = self.tokenizer(prompt, max_length = self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.tokenizer(str(label), max_length = 5, padding = "max_length", truncation=True, return_tensors="pt")
        labels = labels["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        
        return model_inputs

    def __getitem__(self, idx) :
        prompt = self.make_prompt(self.data[idx])
        model_input = self.tokenize(prompt, self.data[idx]['rating'])
        model_input["input_ids"] = model_input["input_ids"].squeeze(0)
        model_input["attention_mask"] = model_input["attention_mask"].squeeze(0)
        return model_input
        
    def __len__(self) :
        return len(self.data)

class TestDataset(Dataset) :
    def __init__(self, data, tokenizer, max_length) :
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__ (self, idx):
        model_output = {}
        prompt = self.make_prompt(self.data[idx])
        tokenized_prompt = self.tokenizer(prompt, max_length = 2048, padding="max_length", truncation=True, return_tensors="pt")
        tokenized_prompt = tokenized_prompt.input_ids
        model_output["prompt"] = prompt
        model_output["input_ids"] = tokenized_prompt
        model_output["label"] = self.data[idx]['rating']
        return model_output
    
    def make_prompt(self, review) :
        user_id = review['user_id']
        movie_title = review['title']
        text = review['text']
        prompt = f"User id : {user_id}\n Target : {movie_title}\n Review text : {text}\n Prompt : According to above information, what star rating would the user {user_id} give? "
        return prompt

    def __len__(self) :
        return len(self.data)
        
def get_dataloader(data_path, tokenizer, max_length, batch_size = 16, p_val = 0.3) : 
    reviews = []
    data_limit = 500000
    with jsonlines.open(data_path) as file :
        test = 0
        for i in file :
            if test > data_limit :
                break
            reviews.append(i)
            test += 1
            
    print(" ================= Data Loaded ================= ")
    r_data, r_test = train_test_split(reviews, test_size=p_val, shuffle=True, random_state=42)
    r_train, r_val = train_test_split(r_data, test_size=p_val, shuffle=False)
    
    train_dataset = CustomDataset(r_train, tokenizer, max_length)
    val_dataset = CustomDataset(r_val, tokenizer, max_length)
    test_dataset = TestDataset(r_test, tokenizer, max_length)

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = 4)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = 4)
    return train_dataloader, val_dataloader, test_dataloader