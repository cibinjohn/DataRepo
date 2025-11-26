for chunks in chunked:
  token_counts = [item['token_count'] for item in chunks]
  print(sum(token_counts))

-------
import pandas as pd
from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

MAX_LEN = 50
OVERLAP = 10
EFFECTIVE_LEN = MAX_LEN - OVERLAP
SEP = "\n"



# Example: Load your data
data_df = pd.read_csv("drive/MyDrive/exl/test_sample_v2.csv", sep=" ")

def group_utterances(df):
    """
    Groups tokens by speaker_id and utt_id.
    
    MODIFICATION: Now properly tokenizes each token individually and counts
    the actual BERT subword tokens, not just the original tokens.
    """
    utterances = []

    for (speaker, utt), group in df.groupby(["speaker_id", "utt_id"], sort=False):
        tokens = group["text"].tolist()
        labels = group["label"].tolist()
        
        # FIXED: Tokenize each token individually to get accurate count
        bert_token_count = 0
        for token in tokens:
            encoded = tokenizer(token, add_special_tokens=False)
            bert_token_count += len(encoded["input_ids"])

        utterances.append({
            "speaker_id": speaker,
            "utt_id": utt,
            "tokens": tokens,  # Original tokens
            "labels": labels,  # Original labels
            "token_count": bert_token_count  # BERT subword count
        })

    return utterances


def split_large_utterance(utt):
    """
    Splits utterances longer than MAX_LEN into smaller chunks.
    
    MODIFICATION: Now properly handles subword tokenization by tracking
    which original tokens map to which BERT tokens.
    """
    original_tokens = utt["tokens"]
    original_labels = utt["labels"]
    chunks = []

    # Tokenize all tokens and track mappings
    all_bert_tokens = []
    all_bert_labels = []
    
    for orig_token, orig_label in zip(original_tokens, original_labels):
        encoded = tokenizer(orig_token, add_special_tokens=False)
        bert_pieces = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        
        # All subword pieces get the same label as their parent token
        for piece in bert_pieces:
            all_bert_tokens.append(piece)
            all_bert_labels.append(orig_label)
    
    # Now split into chunks of MAX_LEN
    start = 0
    while start < len(all_bert_tokens):
        piece_tokens = all_bert_tokens[start:start + MAX_LEN]
        piece_labels = all_bert_labels[start:start + MAX_LEN]

        chunks.append([{
            "speaker_id": utt["speaker_id"],
            "utt_id": utt["utt_id"],
            "tokens": piece_tokens,  # BERT tokens
            "labels": piece_labels,
            "token_count": len(piece_tokens)
        }])

        start += EFFECTIVE_LEN

    return chunks


def create_overlap_chunk(chunk):
    """
    Creates overlap from the last OVERLAP tokens of the previous chunk.
    
    MAJOR MODIFICATION: Skips existing overlap utterances (speaker_id=-1) and only
    uses real utterances to create new overlaps with correct speaker attribution.
    """
    # Flatten all tokens, labels, AND speakers from the chunk
    # IMPORTANT: Skip overlap utterances from previous chunks (speaker_id=-1)
    flat_items = []  # List of (token, label, speaker_id)
    flat_bert_count = 0
    
    for utt in chunk:
        # Skip overlap utterances - we only want real utterances for creating new overlaps
        if utt["speaker_id"] == -1:
            continue
            
        # If this is already a BERT-tokenized utterance (from split_large_utterance),
        # we need to handle it differently
        if any(tok.startswith("##") for tok in utt["tokens"]):
            # Already BERT tokens - keep as is
            for tok, lab in zip(utt["tokens"], utt["labels"]):
                flat_items.append((tok, lab, utt["speaker_id"]))
            flat_bert_count += len(utt["tokens"])
        else:
            # Original tokens - keep them as original tokens but count BERT tokens
            for orig_token, orig_label in zip(utt["tokens"], utt["labels"]):
                encoded = tokenizer(orig_token, add_special_tokens=False)
                bert_token_count = len(encoded["input_ids"])
                
                flat_items.append((orig_token, orig_label, utt["speaker_id"]))
                flat_bert_count += bert_token_count
    
    # Work backwards to find which original tokens make up the last OVERLAP BERT tokens
    overlap_items = []
    bert_count_so_far = 0
    
    for i in range(len(flat_items) - 1, -1, -1):
        token, label, speaker = flat_items[i]
        
        # Count how many BERT tokens this original token produces
        if token.startswith("##"):
            token_bert_count = 1
        else:
            encoded = tokenizer(token, add_special_tokens=False)
            token_bert_count = len(encoded["input_ids"])
        
        if bert_count_so_far + token_bert_count <= OVERLAP:
            overlap_items.insert(0, (token, label, speaker))
            bert_count_so_far += token_bert_count
        else:
            break
    
    # Group consecutive tokens by speaker to create multiple utterances if needed
    overlap_utterances = []
    if overlap_items:
        current_speaker = overlap_items[0][2]
        current_tokens = [overlap_items[0][0]]
        current_labels = [overlap_items[0][1]]
        current_count = 1 if overlap_items[0][0].startswith("##") else len(tokenizer(overlap_items[0][0], add_special_tokens=False)["input_ids"])
        
        for token, label, speaker in overlap_items[1:]:
            if speaker == current_speaker:
                # Same speaker, add to current utterance
                current_tokens.append(token)
                current_labels.append(label)
                token_count = 1 if token.startswith("##") else len(tokenizer(token, add_special_tokens=False)["input_ids"])
                current_count += token_count
            else:
                # Speaker changed, save current and start new
                overlap_utterances.append({
                    "speaker_id": current_speaker,
                    "utt_id": -1,
                    "tokens": current_tokens,
                    "labels": current_labels,
                    "token_count": current_count
                })
                current_speaker = speaker
                current_tokens = [token]
                current_labels = [label]
                current_count = 1 if token.startswith("##") else len(tokenizer(token, add_special_tokens=False)["input_ids"])
        
        # Add the last group
        overlap_utterances.append({
            "speaker_id": current_speaker,
            "utt_id": -1,
            "tokens": current_tokens,
            "labels": current_labels,
            "token_count": current_count
        })
    
    return overlap_utterances if overlap_utterances else []


def chunk_utterances(utterances):
    """
    Chunks utterances into sequences of MAX_LEN tokens with OVERLAP.
    
    MODIFICATION: Updated token counting logic to handle both original
    and BERT-tokenized utterances correctly.
    """
    chunks = []
    current_chunk = []
    current_token_count = 0

    i = 0
    while i < len(utterances):
        utt = utterances[i]

        # Force-split long utterances
        if utt["token_count"] > MAX_LEN:
            # If current chunk exists, save it first
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_token_count = 0
            
            # Add split chunks
            long_chunks = split_large_utterance(utt)
            chunks.extend(long_chunks)
            i += 1
            continue

        # Check if current utterance fits in the chunk
        if current_token_count + utt["token_count"] > MAX_LEN:
            # Save full chunk
            if current_chunk:
                chunks.append(current_chunk)

                # Build next chunk using overlap
                current_chunk = create_overlap_chunk(current_chunk)
                current_token_count = current_chunk[0]["token_count"]
            else:
                # Should never happen, but safe guard
                i += 1
                continue

        # Add utterance normally
        current_chunk.append(utt)
        current_token_count += utt["token_count"]
        i += 1

    # Add the final chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunks_to_string(chunks):
    """
    Converts chunks into annotation format with Agent:/Customer: prefixes.
    
    MODIFICATION: Forces speaker prefix at the start of every chunk, then adds
    prefixes when speaker changes within the chunk.
    """
    output_lines = []

    for chunk_idx, chunk in enumerate(chunks):
        chunk_lines = []
        last_speaker = None
        
        # Debug: print what we're getting
        # print(f"\n=== CHUNK {chunk_idx} ===")
        # for i, utt in enumerate(chunk):
        #     print(f"Utterance {i}: speaker_id={utt['speaker_id']}, tokens={utt['tokens'][:3]}...")
        
        for utt in chunk:
            current_speaker = utt["speaker_id"]
            
            # Add speaker prefix if speaker is valid and (first in chunk OR changed)
            if current_speaker in [0, 1]:
                if last_speaker is None or current_speaker != last_speaker:
                    if current_speaker == 0:
                        chunk_lines.append("Agent: O")
                    elif current_speaker == 1:
                        chunk_lines.append("Customer: O")
                    last_speaker = current_speaker
            else:
                # This shouldn't happen - overlap should have real speaker IDs now
                print(f"WARNING: Found speaker_id={current_speaker} (should be 0 or 1)")
            
            # Add tokens
            for tok, lab in zip(utt["tokens"], utt["labels"]):
                chunk_lines.append(f"{tok} {lab}")
        
        # Add chunk to output
        output_lines.extend(chunk_lines)
        output_lines.append("")  # Blank line between chunks

    return "\n".join(output_lines)


utterances = group_utterances(data_df)
chunked = chunk_utterances(utterances)
annotation_text = chunks_to_string(chunked)
print(annotation_text)


import pandas as pd

# Assuming your dataframe is already loaded as data_df
# Columns: ['speaker_id', 'utt_id', 'text', 'label']

def print_speaker_info(df):
    df = df.copy()
    
    # Ensure data is sorted by utterance and token order
    df = df.sort_values(by=['utt_id']).reset_index(drop=True)
    
    result_tokens = []
    result_utts = []

    current_speaker = None
    token_start = None
    utt_start = None

    for i, row in df.iterrows():
        speaker = row['speaker_id']
        utt = row['utt_id']
        token = row['text']

        if current_speaker is None:
            # First row initialization
            current_speaker = speaker
            token_start = token
            utt_start = utt
            token_end = token
            utt_end = utt
        elif speaker == current_speaker:
            # Same speaker, just update end
            token_end = token
            utt_end = utt
        else:
            # Speaker changed, record previous speaker info
            result_tokens.append(f"{'Agent' if current_speaker==0 else 'Customer'}: {token_start} → {token_end}")
            result_utts.append(f"{'Agent' if current_speaker==0 else 'Customer'}: utt {utt_start} → utt {utt_end}")
            
            # Start new speaker
            current_speaker = speaker
            token_start = token
            token_end = token
            utt_start = utt
            utt_end = utt

    # Add last speaker info
    result_tokens.append(f"{'Agent' if current_speaker==0 else 'Customer'}: {token_start} → {token_end}")
    result_utts.append(f"{'Agent' if current_speaker==0 else 'Customer'}: utt {utt_start} → utt {utt_end}")

    # Print results
    print("=== Speaker vs Tokens ===")
    for line in result_tokens:
        print(line)
    print("\n=== Speaker vs Utterances ===")
    for line in result_utts:
        print(line)

# Call the function
print_speaker_info(data_df)



import pandas as pd

def print_speaker_info_combined(df):
    df = df.copy()
    
    # Sort by utterance ID to maintain sequence
    df = df.sort_values(by=['utt_id']).reset_index(drop=True)
    
    current_speaker = None
    token_start = None
    token_end = None
    utt_start = None
    utt_end = None

    for i, row in df.iterrows():
        speaker = row['speaker_id']
        utt = row['utt_id']
        token = row['text']

        if current_speaker is None:
            # Initialize first block
            current_speaker = speaker
            token_start = token_end = token
            utt_start = utt_end = utt
        elif speaker == current_speaker:
            # Same speaker, extend block
            token_end = token
            utt_end = utt
        else:
            # Speaker changed, print previous block
            print(f"{'Agent' if current_speaker==0 else 'Customer'}: {token_start} → {token_end} : utt {utt_start} → utt {utt_end}")
            
            # Start new block
            current_speaker = speaker
            token_start = token_end = token
            utt_start = utt_end = utt

    # Print last block
    print(f"{'Agent' if current_speaker==0 else 'Customer'}: {token_start} → {token_end} : utt {utt_start} → utt {utt_end}")

# Call the function
print_speaker_info_combined(data_df)


def parse_predictions(prediction_text):
    """
    Parses the prediction text into structured chunks.
    
    Returns:
        List of chunks, where each chunk is a list of (token, label, speaker_id) tuples
    """
    chunks = []
    current_chunk = []
    current_speaker = None
    
    lines = prediction_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Empty line marks end of chunk
        if not line:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_speaker = None
            continue
        
        # Check for speaker prefix
        if line.startswith("Agent: O"):
            current_speaker = 0
            continue
        elif line.startswith("Customer: O"):
            current_speaker = 1
            continue
        
        # Parse token-label pair
        parts = line.split()
        if len(parts) >= 2:
            token = parts[0]
            label = parts[1]
            current_chunk.append((token, label, current_speaker))
    
    # Add last chunk if exists
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def stitch_predictions_prefer_previous(chunks):
    """
    Stitches predictions from multiple chunks, preferring predictions from
    the PREVIOUS chunk for overlapping tokens.
    
    This is the default behavior - once a token has been predicted in a chunk,
    that prediction is kept even if the token appears again in the next chunk's overlap.
    
    Args:
        chunks: List of chunks from parse_predictions()
    
    Returns:
        List of (token, label, speaker_id) tuples for the complete sequence
    """
    if not chunks:
        return []
    
    # Start with the first chunk
    stitched = chunks[0].copy()
    seen_tokens = set()
    
    # Track which tokens we've seen (using position tracking would be better,
    # but since token names are unique in your example, we use token names)
    for token, label, speaker in chunks[0]:
        seen_tokens.add(token)
    
    # Process remaining chunks
    for chunk in chunks[1:]:
        for token, label, speaker in chunk:
            # Only add tokens we haven't seen before (skip overlaps)
            if token not in seen_tokens:
                stitched.append((token, label, speaker))
                seen_tokens.add(token)
    
    return stitched



def stitch_predictions_prefer_next(chunks):
    """
    Stitches predictions from multiple chunks, preferring predictions from
    the NEXT chunk for overlapping tokens.
    
    This means overlap predictions from the later chunk override earlier predictions.
    Useful if you believe the model makes better predictions when tokens appear
    earlier in the sequence (less context decay).
    
    Args:
        chunks: List of chunks from parse_predictions()
    
    Returns:
        List of (token, label, speaker_id) tuples for the complete sequence
    """
    if not chunks:
        return []
    
    # Build a dictionary to track the LAST prediction for each token
    token_predictions = {}
    token_order = []  # Track order of first appearance
    
    for chunk_idx, chunk in enumerate(chunks):
        for token, label, speaker in chunk:
            # Update prediction (later chunks override earlier ones)
            if token not in token_predictions:
                token_order.append(token)
            token_predictions[token] = (label, speaker)
    
    # Reconstruct sequence in original order with updated predictions
    stitched = []
    for token in token_order:
        label, speaker = token_predictions[token]
        stitched.append((token, label, speaker))
    
    return stitched


def format_stitched_output(stitched_predictions):
    """
    Formats stitched predictions back into the original annotation format.
    
    Args:
        stitched_predictions: List of (token, label, speaker_id) tuples
    
    Returns:
        Formatted string in the original annotation format
    """
    output_lines = []
    last_speaker = None
    
    for token, label, speaker in stitched_predictions:
        # Add speaker prefix when speaker changes
        if speaker != last_speaker:
            if speaker == 0:
                output_lines.append("Agent: O")
            elif speaker == 1:
                output_lines.append("Customer: O")
            last_speaker = speaker
        
        # Add token-label pair
        output_lines.append(f"{token} {label}")
    
    return "\n".join(output_lines)


# Parse the predictions
chunks = parse_predictions(annotation_text)

# Option 1: Prefer predictions from previous chunk (default behavior)
stitched_prev = stitch_predictions_prefer_previous(chunks)
output_prev = format_stitched_output(stitched_prev)
print("=== PREFERRING PREVIOUS CHUNK ===")
print(output_prev)


# Option 2: Prefer predictions from next chunk (overlap updates)
stitched_next = stitch_predictions_prefer_next(chunks)
output_next = format_stitched_output(stitched_next)
print("\n=== PREFERRING NEXT CHUNK ===")
print(output_next)
