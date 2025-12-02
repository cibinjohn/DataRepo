from typing import List, Tuple

TokenTriple = Tuple[str, str, object]  # (token, label, speaker_id)

def stitch_predictions_prefer_previous(chunks: List[List[TokenTriple]]) -> List[TokenTriple]:
    """
    Stitch predictions from multiple chunks, preferring predictions from the
    previous chunk for overlapping tokens.

    Args:
        chunks: List of chunks, where each chunk is a list of (token, label, speaker_id).

    Returns:
        A single list of (token, label, speaker_id) for the full sequence.
    """
    if not chunks:
        return []

    # Start stitched result with first chunk (copy to avoid mutating input)
    stitched: List[TokenTriple] = list(chunks[0])

    def max_overlap_len(a: List[TokenTriple], b: List[TokenTriple]) -> int:
        """
        Return the largest k such that the last k tokens of `a` equal the first k tokens of `b`.
        Matching is done on (token, speaker_id) to be more robust against label differences.
        """
        max_k = min(len(a), len(b))
        # We'll test k from max_k down to 1 to find the largest match quickly
        for k in range(max_k, 0, -1):
            # compare last k of a to first k of b
            tail = a[-k:]
            head = b[:k]
            # match by token text and speaker only (ignores labels so previous labels are preferred)
            match = True
            for (t1, _, s1), (t2, _, s2) in zip(tail, head):
                if t1 != t2 or s1 != s2:
                    match = False
                    break
            if match:
                return k
        return 0

    # Process remaining chunks
    for chunk in chunks[1:]:
        if not chunk:
            continue
        k = max_overlap_len(stitched, chunk)
        # If k > 0, we keep the stitched tail (previous predictions) and append only the non-overlapping part
        if k > 0:
            stitched.extend(chunk[k:])  # append remainder
        else:
            # No overlap detected: append full chunk
            stitched.extend(chunk)

    return stitched


# -------------------------
# Example / quick unit test
# -------------------------
if __name__ == "__main__":
    # tokens are (token, label, speaker)
    c1 = [("I", "O", 0), ("love", "O", 0), ("python", "B-LANG", 0), ("very", "O", 0)]
    # chunk2 overlaps on ("python", speaker 0), ("very", speaker 0) and continues
    c2 = [("python", "B-LANG", 0), ("very", "O", 0), ("much", "O", 0)]
    # chunk3 begins with a token identical to earlier but different speaker -> no overlap
    c3 = [("I", "O", 1), ("like", "O", 1)]

    stitched = stitch_predictions_prefer_previous([c1, c2, c3])
    print(stitched)
    # Expected:
    # [
    #  ("I","O",0), ("love","O",0), ("python","B-LANG",0), ("very","O",0),
    #  ("much","O",0), ("I","O",1), ("like","O",1)
    # ]
