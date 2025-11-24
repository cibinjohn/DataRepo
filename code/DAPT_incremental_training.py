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
import gc

# Configuration
class Config:
    model_name = "distilbert-base-uncased"
    max_length = 128  # Reduced from 512 for CPU efficiency
    mlm_probability = 0.15
    batch_size = 4  # Reduced batch size for CPU
    num_epochs = 1
    learning_rate = 5e-5
    save_dir = "model_save_dir"
    eval_steps = 500
    save_steps = 1000
    seed = 42
    gradient_accumulation_steps = 4  # Simulate larger batch size
    max_grad_norm = 1.0
    dataloader_num_workers = 2  # Parallel data loading
    fp16 = False  # CPU doesn't support fp16
    chunk_size = 1000  # For incremental training

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Custom Dataset for text data
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

# Memory-efficient streaming dataset
class StreamingTextDataset(Dataset):
    """Process data in chunks to save memory"""
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

# Evaluation function with reproducible results
def evaluate_perplexity(model, tokenizer, texts, batch_size=4, max_length=128, 
                        mlm_probability=0.15, seed=42, num_workers=2):
    """
    Evaluate perplexity on a list of texts with reproducible results.
    Optimized for CPU with parallel data loading.
    """
    set_seed(seed)
    
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    
    # Use streaming dataset for memory efficiency
    eval_dataset = StreamingTextDataset(texts, tokenizer, max_length)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
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
            num_tokens = (labels != -100).sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity

def save_model(model, tokenizer, save_path):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

def load_model(load_path):
    model = DistilBertForMaskedLM.from_pretrained(load_path)
    tokenizer = DistilBertTokenizer.from_pretrained(load_path)
    print(f"Model loaded from {load_path}")
    return model, tokenizer

def train_dapt_mlm(train_texts, eval_texts=None, use_streaming=True):
    """
    Train DistilBERT with MLM optimized for CPU with limited resources.
    
    Args:
        train_texts: List of training text strings
        eval_texts: Optional list of evaluation text strings
        use_streaming: Use streaming dataset for memory efficiency
    """
    set_seed(Config.seed)
    
    print("Initializing tokenizer and model...")
    tokenizer = DistilBertTokenizer.from_pretrained(Config.model_name)
    model = DistilBertForMaskedLM.from_pretrained(Config.model_name)
    
    print(f"Creating datasets (streaming={use_streaming})...")
    if use_streaming:
        train_dataset = StreamingTextDataset(train_texts, tokenizer, Config.max_length)
        eval_dataset = StreamingTextDataset(eval_texts, tokenizer, Config.max_length) if eval_texts else None
    else:
        train_dataset = TextDataset(train_texts, tokenizer, Config.max_length)
        eval_dataset = TextDataset(eval_texts, tokenizer, Config.max_length) if eval_texts else None
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=Config.mlm_probability
    )
    
    training_args = TrainingArguments(
        output_dir=Config.save_dir,
        overwrite_output_dir=True,
        num_train_epochs=Config.num_epochs,
        per_device_train_batch_size=Config.batch_size,
        per_device_eval_batch_size=Config.batch_size,
        gradient_accumulation_steps=Config.gradient_accumulation_steps,
        learning_rate=Config.learning_rate,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=Config.eval_steps if eval_dataset else None,
        save_steps=Config.save_steps,
        save_total_limit=2,
        logging_steps=100,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        fp16=False,  # CPU doesn't support fp16
        report_to="none",
        seed=Config.seed,
        dataloader_num_workers=Config.dataloader_num_workers,
        max_grad_norm=Config.max_grad_norm,
        dataloader_pin_memory=False,
        no_cuda=True  # Force CPU usage
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    print("Starting DAPT training (CPU mode with optimizations)...")
    print(f"  - Batch size: {Config.batch_size}")
    print(f"  - Gradient accumulation: {Config.gradient_accumulation_steps}")
    print(f"  - Effective batch size: {Config.batch_size * Config.gradient_accumulation_steps}")
    print(f"  - Data workers: {Config.dataloader_num_workers}")
    print(f"  - Max sequence length: {Config.max_length}")
    
    trainer.train()
    
    save_model(model, tokenizer, Config.save_dir)
    
    if eval_dataset:
        perplexity = evaluate_perplexity(
            model=model,
            tokenizer=tokenizer,
            texts=eval_texts,
            batch_size=Config.batch_size,
            max_length=Config.max_length,
            mlm_probability=Config.mlm_probability,
            seed=Config.seed,
            num_workers=Config.dataloader_num_workers
        )
        print(f"\nFinal Evaluation Perplexity: {perplexity:.2f}")
    
    return model, tokenizer

def train_incremental(chunks, checkpoint_dir="model_checkpoints", 
                     chunk_size=1000, resume_from=None):
    """
    Train incrementally on chunks of data to save memory.
    Useful for very large datasets on CPU.
    
    Args:
        chunks: Full list of text data
        checkpoint_dir: Directory to save checkpoints
        chunk_size: Number of texts to process at a time
        resume_from: Path to checkpoint to resume from
    """
    set_seed(Config.seed)
    
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        model, tokenizer = load_model(resume_from)
    else:
        print("Initializing new model...")
        tokenizer = DistilBertTokenizer.from_pretrained(Config.model_name)
        model = DistilBertForMaskedLM.from_pretrained(Config.model_name)
    
    num_chunks = (len(chunks) + chunk_size - 1) // chunk_size
    print(f"\nTraining on {len(chunks)} texts in {num_chunks} chunks of {chunk_size}")
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(chunks))
        chunk_data = chunks[start_idx:end_idx]
        
        print(f"\n{'='*60}")
        print(f"Processing chunk {i+1}/{num_chunks} ({len(chunk_data)} texts)")
        print(f"{'='*60}")
        
        train_dataset = StreamingTextDataset(chunk_data, tokenizer, Config.max_length)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=Config.mlm_probability
        )
        
        chunk_save_dir = f"{checkpoint_dir}/chunk_{i+1}"
        
        training_args = TrainingArguments(
            output_dir=chunk_save_dir,
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=Config.batch_size,
            gradient_accumulation_steps=Config.gradient_accumulation_steps,
            learning_rate=Config.learning_rate,
            save_steps=Config.save_steps,
            save_total_limit=1,
            logging_steps=50,
            fp16=False,
            report_to="none",
            seed=Config.seed,
            dataloader_num_workers=Config.dataloader_num_workers,
            no_cuda=True
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        trainer.train()
        
        # Save checkpoint after each chunk
        save_model(model, tokenizer, chunk_save_dir)
        
        # Clear memory
        del train_dataset, trainer
        gc.collect()
        
        print(f"Chunk {i+1} completed and saved to {chunk_save_dir}")
    
    # Save final model
    final_save_path = f"{checkpoint_dir}/final_model"
    save_model(model, tokenizer, final_save_path)
    print(f"\nIncremental training completed! Final model saved to {final_save_path}")
    
    return model, tokenizer

# Example usage
if __name__ == "__main__":
    # Sample data (replace with your 'chunks' variable)
    chunks = [
        "Machine learning models require large amounts of training data.",
        "Neural networks consist of interconnected layers of neurons.",
        "Deep learning has revolutionized computer vision tasks.",
        "Natural language processing enables computers to understand text.",
        "Transformers use self-attention mechanisms for sequence modeling.",
        "Artificial intelligence systems learn from experience.",
        "Convolutional neural networks excel at image recognition.",
        "Recurrent networks process sequential data effectively.",
    ] * 100  # Simulate larger dataset
    
    print("="*60)
    print("CPU-OPTIMIZED TRAINING OPTIONS")
    print("="*60)
    print("\nOption 1: Standard training with optimizations")
    print("Option 2: Incremental training (for very large datasets)")
    
    # OPTION 1: Standard training with CPU optimizations
    print("\n" + "="*60)
    print("OPTION 1: STANDARD TRAINING")
    print("="*60)
    
    model, tokenizer = train_dapt_mlm(chunks, chunks, use_streaming=True)
    
    # Evaluate
    perplexity = evaluate_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=chunks[:100],  # Evaluate on subset
        batch_size=Config.batch_size,
        max_length=Config.max_length,
        seed=Config.seed,
        num_workers=Config.dataloader_num_workers
    )
    print(f"\nPerplexity: {perplexity:.2f}")
    
    # OPTION 2: Incremental training (uncomment to use)
    # print("\n" + "="*60)
    # print("OPTION 2: INCREMENTAL TRAINING")
    # print("="*60)
    # 
    # model, tokenizer = train_incremental(
    #     chunks=chunks,
    #     checkpoint_dir="model_checkpoints",
    #     chunk_size=500  # Process 500 texts at a time
    # )
    
    print("\n" + "="*60)
    print("TESTING LOADED MODEL")
    print("="*60)
    
    # Load and test
    loaded_model, loaded_tokenizer = load_model(Config.save_dir)
    
    perplexity = evaluate_perplexity(
        model=loaded_model,
        tokenizer=loaded_tokenizer,
        texts=chunks[:100],
        batch_size=Config.batch_size,
        max_length=Config.max_length,
        seed=Config.seed
    )
    print(f"Perplexity (loaded model): {perplexity:.2f}")
    
    print("\n✓ Training completed with CPU optimizations!")
    print(f"✓ Using {Config.dataloader_num_workers} workers for parallel data loading")
    print(f"✓ Gradient accumulation: {Config.gradient_accumulation_steps} steps")
    print(f"✓ Reduced sequence length: {Config.max_length} tokens")
