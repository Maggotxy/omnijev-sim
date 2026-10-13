"""Bounded HTTP clients; credentials come from environment variables."""
import base64
import hashlib
import io
import json
import math
import os
import re
import time
import urllib.error
import urllib.request


class APIError(RuntimeError):
    def __init__(self, status, detail):
        self.status = status
        super().__init__(f"HTTP {status}: {detail}")


def redact(text):
    for name in ("GEMINI_API_KEY", "OMNIJEV_KEY"):
        value = os.environ.get(name)
        if value:
            text = text.replace(value, "[REDACTED]")
    return re.sub(r"projects/[^/` ]+", "projects/[REDACTED]", text)


def png_bytes(frame):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, "PNG")
    return buf.getvalue()


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def validate_plan(value, candidates):
    obj, target = value.get("object"), value.get("target")
    if obj == "finish":
        return value
    if obj not in candidates or target not in candidates or obj == target:
        raise ValueError("Plan must name two distinct objects in the perception catalog")
    return value


def checked_choice(answer):
    if answer.get("valid") is False or not isinstance(answer.get("choice"), str):
        raise ValueError("Model abstained or returned an invalid choice")
    probability = float(answer.get("probabilities", {}).get(answer["choice"], 0))
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Choice probability must be finite and between zero and one")
    return answer["choice"], probability


class Client:
    def __init__(self, trace=None):
        self.trace, self.records = trace, []

    def post(self, url, body, headers, purpose, frame=None):
        record = {"purpose": purpose, "requested_model": self.model,
                  "image_sha256": hashlib.sha256(png_bytes(frame)).hexdigest() if frame is not None else None}
        start = time.monotonic()
        print(f"API {type(self).__name__} {purpose}", flush=True)
        try:
            request = urllib.request.Request(url, json.dumps(body).encode(),
                                             {"Content-Type": "application/json", **headers})
            try:
                with urllib.request.urlopen(request, timeout=25) as response:
                    value = json.loads(response.read())
                    record.update(http_status=response.status,
                                  served_model=response.headers.get("X-Jev-Model") or value.get("modelVersion"))
            except urllib.error.HTTPError as exc:
                raise APIError(exc.code, redact(exc.read().decode(errors="replace"))[:1600]) from None
            record["response"] = value
            return value
        except Exception as exc:
            record.update(error=redact(str(exc)), http_status=getattr(exc, "status", 0))
            raise
        finally:
            record["latency_s"] = round(time.monotonic() - start, 4)
            self.records.append(record)
            if self.trace:
                with open(self.trace, "a") as stream:
                    stream.write(json.dumps(record) + "\n")
            print(f"API result {record.get('http_status')} {record['latency_s']:.2f}s", flush=True)


class Gemini(Client):
    def __init__(self, trace=None):
        super().__init__(trace)
        self.base = os.environ.get("GEMINI_BASE", "https://generativelanguage.googleapis.com").rstrip("/")
        self.model = os.environ.get("GEMINI_MODEL", "gemini-robotics-er-2-preview")
        self.key = os.environ.get("GEMINI_API_KEY", "")
        self.protocol = os.environ.get("GEMINI_PROTOCOL", "generateContent")
        if "streaming" in self.model:
            raise ValueError("The streaming Robotics endpoint requires a Live API adapter")
        if self.protocol not in ("generateContent", "interactions"):
            raise ValueError("GEMINI_PROTOCOL must be generateContent or interactions")

    def ask(self, prompt, frame=None, purpose="decision"):
        if not self.key:
            raise APIError(0, "GEMINI_API_KEY is required")
        if self.protocol == "generateContent":
            parts = [{"text": prompt}]
            if frame is not None:
                parts.insert(0, {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(png_bytes(frame)).decode()}})
            body = {"contents": [{"role": "user", "parts": parts}],
                    "generationConfig": {"temperature": 0, "maxOutputTokens": 2048}}
            response = self.post(self.base + "/v1beta/models/" + self.model + ":generateContent",
                                 body, {"x-goog-api-key": self.key}, purpose, frame)
            text = "".join(part.get("text", "") for candidate in response.get("candidates", [])
                           for part in candidate.get("content", {}).get("parts", []) if not part.get("thought"))
        else:
            parts = [{"type": "text", "text": prompt}]
            if frame is not None:
                parts.insert(0, {"type": "image", "mime_type": "image/png", "data": base64.b64encode(png_bytes(frame)).decode()})
            response = self.post(self.base + "/v1beta/interactions", {"model": self.model, "input": parts},
                                 {"x-goog-api-key": self.key}, purpose, frame)
            text = response.get("output_text") or "".join(x.get("text", "") for x in response.get("outputs", []) if x.get("type") == "text")
        if not text:
            raise ValueError("Gemini returned no text")
        return text

    def health(self):
        return self.ask("Reply with OK.", purpose="preflight")

    def plan(self, frame, task, candidates, completed):
        prompt = (f"Robot camera view. Task: {task}. Known labels from shared simulated perception: {candidates}. "
                  f"Already placed: {completed}. Choose the next pick-and-place subtask. "
                  'Return only JSON: {"object":"exact label or finish","target":"exact destination label","reason":"short explanation"}. '
                  "Use finish only if the full instruction appears completed.")
        return validate_plan(parse_json(self.ask(prompt, frame, "plan")), candidates)

    def gate(self, frame, context):
        prompt = (f"Robot camera view. {context}. Decide whether to execute the proposed skill segment now. "
                  'Return only JSON: {"action":"proceed|wait|replan","reason":"short explanation"}. '
                  "proceed executes the segment; wait holds this pose; replan aborts this attempt and requests a new plan.")
        value = parse_json(self.ask(prompt, frame, "gate"))
        if value.get("action") not in ("proceed", "wait", "replan"):
            raise ValueError("Invalid Gemini gate action")
        return value


class Omni(Client):
    def __init__(self, trace=None):
        super().__init__(trace)
        self.base = os.environ.get("OMNIJEV_BASE", "http://39.106.166.185:6004").rstrip("/")
        self.model = os.environ.get("OMNIJEV_MODEL", "omnijev-0.8b-kk")
        self.key = os.environ.get("OMNIJEV_KEY", "")

    def ask(self, frame, questions, purpose):
        if not self.key:
            raise APIError(0, "OMNIJEV_KEY is required")
        body = {"model": self.model, "questions": questions}
        if frame is not None:
            body["image_b64"] = base64.b64encode(png_bytes(frame)).decode()
        return self.post(self.base + "/v1/systemone", body,
                         {"Authorization": "Bearer " + self.key}, purpose, frame)

    def health(self):
        return self.ask(None, {"ok": {"type": "noul", "instructions": "This is a connectivity test."}}, "preflight")

    def plan(self, frame, task, candidates, completed):
        choices = {x: x for x in candidates}
        destination = self.ask(frame, {
            "target": {"type": "choice", "instructions": f"Robot camera view. Instruction: {task}. Where should the picked object be placed?", "criteria": choices}}, "plan_target")
        target, _ = checked_choice(destination["target"])
        if target not in candidates:
            raise ValueError("Destination outside perception catalog")
        # Enumerate executable sub-tasks, with the already selected receiver
        # excluded as a source. This is the same distinct-object tool contract
        # enforced for Gemini's structured output.
        response = self.ask(frame, {
            "object": {"type": "choice", "instructions": f"Robot camera view. Instruction: {task}. Already placed: {completed}. Which sub-task should the robot carry out next?",
                       "criteria": {**{x: f"pick up the {x} and put it in the {target}" for x in candidates if x != target},
                                    "finish": "finish the task because all requested objects have been placed"}}}, "plan_object")
        obj, _ = checked_choice(response["object"])
        return validate_plan({"object": obj, "target": target,
                              "probabilities": {**destination, **response}}, candidates)

    def gate(self, frame, context):
        response = self.ask(frame, {
            "action": {"type": "choice", "instructions": f"Robot camera view. {context}. What should the controller do next?", "criteria": {
                "proceed": "execute the proposed next skill segment now",
                "wait": "hold the current pose and wait",
                "replan": "abort this attempt and request a new task plan"}},
            "holding": {"type": "noul", "instructions": "The robot gripper is holding an object."},
            "done": {"type": "noul", "instructions": "The full instruction has been completed: " + context.split("; proposed=")[0]}}, "gate")
        action, probability = checked_choice(response["action"])
        if action not in ("proceed", "wait", "replan"):
            raise ValueError("Invalid OmniJev gate action")
        effective = "wait" if action == "proceed" and probability < 0.5 else action
        return {"action": effective, "choice": action, "probability": probability, "raw": response}
