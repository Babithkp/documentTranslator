def group_words_into_lines(words, y_threshold=12):
    words = sorted(words, key=lambda w: w["y1"])
    lines = []

    for word in words:
        added = False

        for line in lines:
            avg_y = sum(w["y1"] for w in line) / len(line)

            if abs(word["y1"] - avg_y) <= y_threshold:
                line.append(word)
                added = True
                break

        if not added:
            lines.append([word])

    return lines


def merge_line(words):
    words = sorted(words, key=lambda w: w["x1"])

    text = " ".join(w["text"] for w in words)

    x1 = min(w["x1"] for w in words)
    y1 = min(w["y1"] for w in words)
    x2 = max(w["x2"] for w in words)
    y2 = max(w["y2"] for w in words)

    normalized_words = []

    for w in words:
        normalized_words.append({
            "text": w["text"],
            "bbox": [
                w["x1"],
                w["y1"],
                w["x2"],
                w["y2"]
            ]
        })

    return {
        "text": text,
        "bbox": [x1, y1, x2, y2],
        "words": normalized_words
    }