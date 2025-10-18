import asyncio
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple, AsyncIterator

from langchain.schema import AIMessage
from playwright.async_api import async_playwright, Page

from google import genai
from google.genai import types
from google.genai.types import Content, Part

from ...models import ModelConfig
from ...utils.types import AgentSettings
from steel import Steel


SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900


def _denormalize_x(x: int, screen_width: int) -> int:
    return int(x / 1000 * screen_width)


def _denormalize_y(y: int, screen_height: int) -> int:
    return int(y / 1000 * screen_height)


async def _execute_function_calls(
    candidate: Any,
    page: Page,
    screen_width: int,
    screen_height: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    results: List[Tuple[str, Dict[str, Any]]] = []
    function_calls: List[Any] = []
    if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
        for part in candidate.content.parts:
            if getattr(part, "function_call", None):
                function_calls.append(part.function_call)

    for function_call in function_calls:
        fname = getattr(function_call, "name", "")
        args = getattr(function_call, "args", {}) or {}
        action_result: Dict[str, Any] = {}

        try:
            if fname == "open_web_browser":
                pass
            elif fname == "navigate":
                url = args.get("url")
                if not url:
                    raise ValueError("Missing 'url' for navigate")
                await page.goto(url, wait_until="domcontentloaded")
            elif fname == "go_back":
                await page.go_back()
            elif fname == "go_forward":
                await page.go_forward()
            elif fname == "search":
                await page.goto("https://www.google.com", wait_until="domcontentloaded")
            elif fname == "click_at":
                x = _denormalize_x(int(args["x"]), screen_width)
                y = _denormalize_y(int(args["y"]), screen_height)
                await page.mouse.click(x, y)
            elif fname == "hover_at":
                x = _denormalize_x(int(args["x"]), screen_width)
                y = _denormalize_y(int(args["y"]), screen_height)
                await page.mouse.move(x, y)
            elif fname == "type_text_at":
                x = _denormalize_x(int(args["x"]), screen_width)
                y = _denormalize_y(int(args["y"]), screen_height)
                text = args.get("text", "")
                press_enter = bool(args.get("press_enter", True))
                clear_before_typing = bool(args.get("clear_before_typing", True))
                await page.mouse.click(x, y)
                if clear_before_typing:
                    await page.keyboard.press("Control+A")
                    await page.keyboard.press("Backspace")
                if text:
                    await page.keyboard.type(text)
                if press_enter:
                    await page.keyboard.press("Enter")
            elif fname == "key_combination":
                keys = args.get("keys")
                if not keys:
                    raise ValueError("Missing 'keys' for key_combination")
                await page.keyboard.press(keys)
            elif fname == "scroll_document":
                direction = (args.get("direction") or "down").lower()
                delta = 800 if direction in ("down", "right") else -800
                if direction in ("left", "right"):
                    await page.mouse.wheel(delta, 0)
                else:
                    await page.mouse.wheel(0, delta)
            elif fname == "scroll_at":
                x = _denormalize_x(int(args["x"]), screen_width)
                y = _denormalize_y(int(args["y"]), screen_height)
                direction = (args.get("direction") or "down").lower()
                magnitude = int(args.get("magnitude", 800))
                await page.mouse.move(x, y)
                delta = magnitude if direction in ("down", "right") else -magnitude
                if direction in ("left", "right"):
                    await page.mouse.wheel(delta, 0)
                else:
                    await page.mouse.wheel(0, delta)
            elif fname == "drag_and_drop":
                x = _denormalize_x(int(args["x"]), screen_width)
                y = _denormalize_y(int(args["y"]), screen_height)
                dx = _denormalize_x(int(args["destination_x"]), screen_width)
                dy = _denormalize_y(int(args["destination_y"]), screen_height)
                await page.mouse.move(x, y)
                await page.mouse.down()
                await page.mouse.move(dx, dy, steps=10)
                await page.mouse.up()
            elif fname == "wait_5_seconds":
                await asyncio.sleep(5)
            else:
                action_result["warning"] = f"Unimplemented or custom function: {fname}"

            try:
                await page.wait_for_load_state(state="domcontentloaded", timeout=5000)
            except Exception:
                pass
            await page.wait_for_timeout(300)

        except Exception as e:
            action_result = {"error": str(e)}

        results.append((fname, action_result))

    return results


async def _get_function_responses(page: Page, results: List[Tuple[str, Dict[str, Any]]]) -> List[types.FunctionResponse]:
    screenshot_bytes = await page.screenshot(type="png")
    current_url = page.url
    function_responses: List[types.FunctionResponse] = []
    for name, result in results:
        response_data: Dict[str, Any] = {"url": current_url}
        response_data.update(result or {})
        function_responses.append(
            types.FunctionResponse(
                name=name,
                response=response_data,
                parts=[
                    types.FunctionResponsePart(
                        inline_data=types.FunctionResponseBlob(
                            mime_type="image/png", data=screenshot_bytes
                        )
                    )
                ],
            )
        )
    return function_responses


async def gemini_computer_use(
    model_config: ModelConfig,
    agent_settings: AgentSettings,
    history: List[Mapping[str, Any]],
    session_id: str,
    cancel_event: Optional[asyncio.Event] = None,
) -> AsyncIterator[str]:
    google_api_key = model_config.api_key or os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        yield AIMessage(content="Missing GOOGLE_API_KEY; cannot start Gemini Computer Use.")
        return

    client = genai.Client(api_key=google_api_key)

    width = SCREEN_WIDTH
    height = SCREEN_HEIGHT

    STEEL_API_KEY = os.getenv("STEEL_API_KEY")
    STEEL_CONNECT_URL = os.getenv("STEEL_CONNECT_URL")
    STEEL_API_URL = os.getenv("STEEL_API_URL")
    steel_client = Steel(steel_api_key=STEEL_API_KEY, base_url=STEEL_API_URL)
    session = steel_client.sessions.retrieve(session_id)

    yield AIMessage(content=f"Initializing browser for session {session.id}...")

    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(
            f"{STEEL_CONNECT_URL}?apiKey={STEEL_API_KEY}&sessionId={session.id}"
        )
        context = browser.contexts[0]
        page = context.pages[0]
        await page.set_viewport_size({"width": width, "height": height})

        if not page.url or page.url == "about:blank":
            try:
                await page.goto("https://www.google.com", wait_until="domcontentloaded")
            except Exception:
                pass

        contents: List[Content] = []
        for m in history:
            role = m.get("role", "user")
            text = m.get("content", "")
            if not isinstance(text, str):
                continue
            contents.append(Content(role=("user" if role == "user" else "model"), parts=[Part(text=text)]))

        screenshot_bytes = await page.screenshot(type="png")
        contents.append(
            Content(
                role="user",
                parts=[
                    Part(text=history[-1].get("content", "")),
                    Part.from_bytes(data=screenshot_bytes, mime_type="image/png"),
                ],
            )
        )

        generate_config = types.GenerateContentConfig(
            tools=[
                types.Tool(
                    computer_use=types.ComputerUse(
                        environment=types.Environment.ENVIRONMENT_BROWSER,
                    )
                )
            ],
        )

        turn_limit = agent_settings.steps or 8
        for turn_index in range(turn_limit):
            if cancel_event and cancel_event.is_set():
                break

            yield AIMessage(content=f"Thinking (turn {turn_index + 1}/{turn_limit})...")

            response = client.models.generate_content(
                model="gemini-2.5-computer-use-preview-10-2025",
                contents=contents,
                config=generate_config,
            )

            if not response or not getattr(response, "candidates", None):
                yield AIMessage(content="No response from model; stopping.")
                break

            candidate = response.candidates[0]
            contents.append(candidate.content)

            has_function_calls = False
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                for part in candidate.content.parts:
                    if getattr(part, "function_call", None):
                        has_function_calls = True
                        args = getattr(part.function_call, "args", {}) or {}
                        safety_decision = args.get("safety_decision")
                        if isinstance(safety_decision, dict) and safety_decision.get("decision") == "require_confirmation":
                            explanation = safety_decision.get("explanation") or "Action requires confirmation."
                            yield AIMessage(content=f"⏸️ CONFIRMATION REQUIRED: {explanation}")
                            return

            if not has_function_calls:
                text_parts: List[str] = []
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    for part in candidate.content.parts:
                        if getattr(part, "text", None):
                            text_parts.append(part.text)
                final_text = " ".join(tp for tp in text_parts if tp).strip()
                if final_text:
                    yield AIMessage(content=final_text)
                break

            yield AIMessage(content="Executing actions...")
            results = await _execute_function_calls(candidate, page, width, height)

            function_responses = await _get_function_responses(page, results)
            contents.append(
                Content(role="user", parts=[Part(function_response=fr) for fr in function_responses])
            )

        yield AIMessage(content="Done.")
