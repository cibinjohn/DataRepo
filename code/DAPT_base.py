model_save_dir = "drive/MyDrive/exl/DAPT/"

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
import numpy as np
from pathlib import Path
import math
import random

class Config:
    model_name = "distilbert-base-uncased"
    max_length = 512
    mlm_probability = 0.15
    batch_size = 8
    num_epochs = 1
    learning_rate = 5e-5
    save_dir = model_save_dir
    eval_steps = 500
    save_steps = 1000
    seed = 42


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
        }


def evaluate_perplexity(model, tokenizer, texts, batch_size=8, max_length=512, 
                        mlm_probability=0.15, seed=42):
    """
    Evaluate perplexity on a list of texts with reproducible results.
    
    Args:
        model: The DistilBERT model for masked language modeling
        tokenizer: The tokenizer corresponding to the model
        texts: List of text strings to evaluate on
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        mlm_probability: Probability of masking tokens (default: 0.15)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        perplexity: The perplexity score
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Create dataset
    eval_dataset = TextDataset(texts, tokenizer, max_length)
    
    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability
    )
    
    # Create dataloader
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False  # Don't shuffle for consistent results
    )
    
    # Calculate perplexity
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            # Count non-padded tokens
            num_tokens = (labels != -100).sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity

# Save model and tokenizer
def save_model(model, tokenizer, save_path):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

# Load model and tokenizer
def load_model(load_path):
    model = DistilBertForMaskedLM.from_pretrained(load_path)
    tokenizer = DistilBertTokenizer.from_pretrained(load_path)
    print(f"Model loaded from {load_path}")
    return model, tokenizer

# Main training function
def train_dapt_mlm(train_texts, eval_texts=None):
    """
    Train DistilBERT with Masked Language Modeling for domain adaptation.
    
    Args:
        train_texts: List of training text strings
        eval_texts: Optional list of evaluation text strings
    
    Returns:
        model: Trained model
        tokenizer: Tokenizer
    """
    # Set seed for reproducibility
    set_seed(Config.seed)
    
    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(Config.model_name)
    model = DistilBertForMaskedLM.from_pretrained(Config.model_name)
    
    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, Config.max_length)
    eval_dataset = TextDataset(eval_texts, tokenizer, Config.max_length) if eval_texts else None
    
    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=Config.mlm_probability
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=Config.save_dir,
        overwrite_output_dir=True,
        num_train_epochs=Config.num_epochs,
        per_device_train_batch_size=Config.batch_size,
        per_device_eval_batch_size=Config.batch_size,
        learning_rate=Config.learning_rate,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=Config.eval_steps if eval_dataset else None,
        save_steps=Config.save_steps,
        save_total_limit=2,
        logging_steps=100,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=Config.seed
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting DAPT training...")
    trainer.train()
    
    # Save final model
    save_model(model, tokenizer, Config.save_dir)
    
    # Evaluate perplexity using the evaluation function
    if eval_dataset:
        perplexity = evaluate_perplexity(
            model=model,
            tokenizer=tokenizer,
            texts=eval_texts,
            batch_size=Config.batch_size,
            max_length=Config.max_length,
            mlm_probability=Config.mlm_probability,
            seed=Config.seed
        )
        print(f"\nFinal Evaluation Perplexity: {perplexity:.2f}")
    
    return model, tokenizer



model, tokenizer = train_dapt_mlm(chunks, chunks)


# Evaluate immediately after training
perplexity_before_save = evaluate_perplexity(
    model=model,
    tokenizer=tokenizer,
    texts=chunks,
    batch_size=Config.batch_size,
    max_length=Config.max_length,
    mlm_probability=Config.mlm_probability,
    seed=Config.seed
)

print(f"Perplexity (before saving): {perplexity_before_save:.2f}")


# Load saved model
loaded_model, loaded_tokenizer = load_model(Config.save_dir)

# Evaluate loaded model (should give same result)
perplexity_after_load = evaluate_perplexity(
    model=loaded_model,
    tokenizer=loaded_tokenizer,
    texts=chunks,
    batch_size=Config.batch_size,
    max_length=Config.max_length,
    mlm_probability=Config.mlm_probability,
    seed=Config.seed
)
print(f"Perplexity (after loading): {perplexity_after_load:.2f}")


