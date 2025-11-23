from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def load_token_label_file(path):
    words, labels = [], []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token, label = line.split()
            words.append(token)
            labels.append(label)

    return words, labels

def chunk_by_subword_tokens(words, labels, max_len=512, overlap=112):
    """
    Preserves original words + labels, but splits based on DistilBERT subword length.
    """
    assert len(words) == len(labels)

    chunks = []
    current_chunk = []
    current_token_count = 0

    # Step size in subword tokens
    step = max_len - overlap

    # Precompute subword lengths for each word
    subword_lens = [len(tokenizer.tokenize(w)) for w in words]

    print("subword_lens : ",subword_lens)

    i = 0
    while i < len(words):
        w = words[i]
        l = labels[i]
        sw_len = subword_lens[i]

        # If adding this word exceeds 512 subword tokens → finalize chunk
        if current_token_count + sw_len > max_len:
            chunks.append(current_chunk)

            # Move the window forward based on subword token count
            token_sum = 0
            j = 0
            # Find the smallest prefix of current_chunk whose subword tokens = step
            while j < len(current_chunk) and token_sum < step:
                token_sum += len(tokenizer.tokenize(current_chunk[j][0]))
                j += 1

            # new starting index:
            # we drop the first j words of the previous chunk
            new_start_word_index = i - (len(current_chunk) - j)
            i = max(new_start_word_index, 0)

            # reset
            current_chunk = []
            current_token_count = 0
            continue

        # Add the word
        current_chunk.append((w, l))
        current_token_count += sw_len
        i += 1

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks



def format_chunks(chunks):
    """
    Output format must be:
      token label
      token label

      token label
      ...
    """
    out = ""
    for chunk in chunks:
        for w, l in chunk:
            out += f"{w} {l}\n"
        out += "\n"
    return out

input_file_loc="drive/MyDrive/EXL/test_sample.txt"
output_file_loc="drive/MyDrive/EXL/test_sample_out.txt"


words, labels = load_token_label_file(input_file_loc)
chunks = chunk_by_subword_tokens(words, labels, max_len=200, overlap=50)

result = format_chunks(chunks)
print(result)
