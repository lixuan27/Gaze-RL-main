"""Answer image and video questions with a trained checkpoint.

The trained policy keeps the architecture of its backbone, so inference is plain
generation; the landing field is only used during training.
"""

from __future__ import annotations

import argparse
import re
from typing import Sequence

import torch

PROMPTS = {
    "en": (
        "You are a multimodal reasoning assistant. First give concise reasoning inside <think>, "
        "then give the final answer inside <answer>. For multiple-choice questions the answer must "
        "be the option letter alone. Your output must be exactly <think>...</think><answer>...</answer>."
    ),
    "zh": "你是多模态推理助手。请先在<think>中给出简洁推理，再在<answer>中给出最终答案。输出必须严格为<think>...</think><answer>...</answer>。",
}

_ANSWER = re.compile(r"<answer>([\s\S]*?)</answer>", re.IGNORECASE)


def extract_answer(text: str) -> str:
    match = _ANSWER.search(text)
    if match:
        return match.group(1).strip()
    return re.sub(r"</?(think|answer)>", "", text, flags=re.IGNORECASE).strip()


def sample_frames(path: str, fps: float = 0.5, max_frames: int = 8) -> list:
    """Decode a video at roughly ``fps`` frames per second, keeping at most ``max_frames``."""
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        native = float(stream.average_rate or 0) or 1.0
        step = max(1, round(native / fps)) if fps > 0 else 1
        picked = [frame.to_image() for i, frame in enumerate(container.decode(stream)) if i % step == 0]
    if len(picked) > max_frames:
        idx = torch.linspace(0, len(picked) - 1, max_frames).round().long().tolist()
        picked = [picked[i] for i in idx]
    return picked


class GazeRLModel:
    """Greedy-decoding wrapper around a checkpoint and its processor.

    ``model("question", images=[...])`` or ``model("question", video=frames)``
    returns the full response and the extracted answer.
    """

    def __init__(self, model: str, prompt: str = "en", dtype: torch.dtype = torch.bfloat16, device_map: str = "auto"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model, dtype=dtype, device_map=device_map, trust_remote_code=True
        ).eval()
        self.system = PROMPTS[prompt]

    def _inputs(self, question: str, images: Sequence | None, video: list | None):
        content = [{"type": "image"} for _ in images or []]
        if video is not None:
            content.append({"type": "video"})
        content.append({"type": "text", "text": question})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system}]},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        kwargs = {"text": [text], "return_tensors": "pt"}
        if images:
            kwargs["images"] = list(images)
        if video is not None:
            kwargs["videos"] = [video]
        return self.processor(**kwargs).to(self.model.device)

    @torch.no_grad()
    def __call__(self, question: str, images: Sequence | None = None, video: list | None = None, max_new_tokens: int = 512) -> dict:
        inputs = self._inputs(question, images, video)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.processor.batch_decode(output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]
        return {"response": text, "answer": extract_answer(text)}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Answer an image or video question with a Gaze-RL checkpoint.")
    parser.add_argument("--model", required=True, help="Hugging Face id or local path of the checkpoint")
    parser.add_argument("--question", required=True)
    parser.add_argument("--image", action="append", default=[], help="image path, may be repeated")
    parser.add_argument("--video", help="video path")
    parser.add_argument("--fps", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="en")
    args = parser.parse_args(argv)

    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in args.image]
    video = sample_frames(args.video, args.fps, args.max_frames) if args.video else None
    result = GazeRLModel(args.model, prompt=args.prompt)(args.question, images, video, args.max_new_tokens)
    print(result["response"])
    print(f"\nAnswer: {result['answer']}")


if __name__ == "__main__":
    main()
