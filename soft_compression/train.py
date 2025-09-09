import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoConfig
from scaledown_model import ScaleDownCompressor, ScaleDownGenerator, ScaleDownCompressorConfig
import torch.nn.functional as F

class QADataset(Dataset):
    """
    A sample dataset for training ScaleDown.
    In a real scenario, this would load queries, documents, and teacher-generated answers.
    """
    def __init__(self, tokenizer, num_samples=1000, max_length=512):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length
        self.num_mem_tokens = 8
        self.num_docs = 5

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Dummy data
        query = "Which films earned Ken Loach the Palme d'Or?"
        documents = ["Ken Loach won the Palme d'Or for The Wind That Shakes the Barley and I, Daniel Blake."] * self.num_docs
        teacher_answer = "The Wind That Shakes the Barley and I, Daniel Blake."
        teacher_rerank_scores = torch.randn(self.num_docs) # Dummy scores

        # Prepare compressor inputs
        compressor_inputs = []
        for doc in documents:
            # [CLS] Query [SEP] Document [MEM]...[MEM] [RR] [SEP]
            text = f"{self.tokenizer.cls_token}{query}{self.tokenizer.sep_token}{doc}"
            mem_tokens = ''.join([f"<MEM{i}>" for i in range(self.num_mem_tokens)])
            rr_token = "<RR>"
            text_with_special_tokens = f"{text}{mem_tokens}{rr_token}{self.tokenizer.sep_token}"
            
            encoding = self.tokenizer(
                text_with_special_tokens,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoding.input_ids.squeeze(0)
            
            # Find indices of special tokens
            mem_indices = torch.tensor([i for i, t_id in enumerate(input_ids) if t_id in self.tokenizer.convert_tokens_to_ids([f"<MEM{i}>" for i in range(self.num_mem_tokens)])])
            rr_index = torch.tensor([i for i, t_id in enumerate(input_ids) if t_id == self.tokenizer.convert_tokens_to_ids("<RR>")][0])

            compressor_inputs.append({
                "input_ids": input_ids,
                "attention_mask": encoding.attention_mask.squeeze(0),
                "memory_token_indices": mem_indices,
                "rerank_token_index": rr_index,
            })

        # Prepare generator inputs
        generator_input = self.tokenizer(query, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.tokenizer(teacher_answer, max_length=128, padding="max_length", truncation=True, return_tensors="pt").input_ids
        
        # Flatten compressor inputs for batching
        final_input = {
            "compressor_input_ids": torch.stack([x['input_ids'] for x in compressor_inputs]),
            "compressor_attention_mask": torch.stack([x['attention_mask'] for x in compressor_inputs]),
            "compressor_memory_token_indices": torch.stack([x['memory_token_indices'] for x in compressor_inputs]),
            "compressor_rerank_token_index": torch.stack([x['rerank_token_index'] for x in compressor_inputs]),
            "generator_input_ids": generator_input.input_ids.squeeze(0),
            "generator_attention_mask": generator_input.attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
            "teacher_rerank_scores": teacher_rerank_scores,
        }

        return final_input

class ScaleDownTrainer(Trainer):
    def __init__(self, compressor, generator, rerank_lambda=0.05, **kwargs):
        # We pass None for model because we handle it manually
        super().__init__(model=None, **kwargs)
        self.compressor = compressor
        self.generator = generator
        self.rerank_lambda = rerank_lambda

    def compute_loss(self, model, inputs, return_outputs=False):
        # Unpack inputs
        compressor_input_ids = inputs.pop("compressor_input_ids")
        compressor_attention_mask = inputs.pop("compressor_attention_mask")
        compressor_memory_token_indices = inputs.pop("compressor_memory_token_indices")
        compressor_rerank_token_index = inputs.pop("compressor_rerank_token_index")
        generator_input_ids = inputs.pop("generator_input_ids")
        generator_attention_mask = inputs.pop("generator_attention_mask")
        labels = inputs.pop("labels")
        teacher_rerank_scores = inputs.pop("teacher_rerank_scores")

        batch_size, num_docs, seq_len = compressor_input_ids.shape
        
        # Reshape for compressor
        compressor_input_ids = compressor_input_ids.view(-1, seq_len)
        compressor_attention_mask = compressor_attention_mask.view(-1, seq_len)
        compressor_memory_token_indices = compressor_memory_token_indices.view(-1, self.train_dataset.num_mem_tokens)
        compressor_rerank_token_index = compressor_rerank_token_index.view(-1, 1)


        # 1. Get compressed embeddings and rerank scores
        compressed_embeddings, rerank_scores = self.compressor(
            input_ids=compressor_input_ids,
            attention_mask=compressor_attention_mask,
            memory_token_indices=compressor_memory_token_indices,
            rerank_token_index=compressor_rerank_token_index
        )
        
        # Reshape back
        compressed_embeddings = compressed_embeddings.view(batch_size, num_docs * self.train_dataset.num_mem_tokens, -1)
        rerank_scores = rerank_scores.view(batch_size, num_docs)


        # 2. Get generator outputs
        generator_outputs = self.generator(
            input_ids=generator_input_ids,
            attention_mask=generator_attention_mask,
            compressed_embeddings=compressed_embeddings,
            labels=labels
        )
        gen_loss = generator_outputs.loss

        # 3. Compute reranking loss
        rerank_loss = F.mse_loss(rerank_scores, teacher_rerank_scores)

        # 4. Total loss
        total_loss = gen_loss + self.rerank_lambda * rerank_loss

        return (total_loss, {"gen_loss": gen_loss, "rerank_loss": rerank_loss}) if return_outputs else total_loss


def main():
    # Model names
    compressor_model_name = "distilbert-base-uncased" # Smaller model for ScaleDown-llama
    generator_model_name = "gpt2"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
    tokenizer.add_special_tokens({
        'cls_token': '[CLS]',
        'sep_token': '[SEP]',
        'pad_token': '[PAD]',
        'additional_special_tokens': [f'<MEM{i}>' for i in range(8)] + ['<RR>']
    })
    
    # Configs
    compressor_config = ScaleDownCompressorConfig(
        compressor_name_or_path=compressor_model_name,
        use_n_layers=False, # Example of ScaleDown-llama
        generator_hidden_size=AutoConfig.from_pretrained(generator_model_name).hidden_size,
        add_reranking_head=True
    )
    generator_config = AutoConfig.from_pretrained(generator_model_name)


    # Models
    compressor = ScaleDownCompressor(compressor_config)
    generator = ScaleDownGenerator(generator_config)
    generator.generator.resize_token_embeddings(len(tokenizer))


    # Dataset
    train_dataset = QADataset(tokenizer)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./scaledown_results",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=100,
        report_to="none"
    )

    # Trainer
    trainer = ScaleDownTrainer(
        compressor=compressor,
        generator=generator,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save models
    compressor.save_pretrained("./scaledown_compressor_final")
    generator.save_pretrained("./scaledown_generator_final")
    tokenizer.save_pretrained("./scaledown_compressor_final")


if __name__ == "__main__":
    main()

