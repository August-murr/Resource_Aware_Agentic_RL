# Continuous Generation Documentation

This documentation covers the continuous generation feature for VERL, which enables agentic workflows with interval-based generation, budget checking, and context window management.

## Overview

Continuous generation allows the agent loop to fill the entire context window with generation intervals, checking budgets and processing tools along the way. Unlike standard rollout mode where generation stops after `max_response_length` tokens, continuous generation continues until hitting the context window limit (`max_model_len`) or budget constraints.

## Key Concepts

### Context Window Management

In continuous generation mode, `max_model_len` is the **fundamental limit** for all rollouts. It represents the total context window size (prompt + all generated tokens).

```bash
actor_rollout_ref.rollout.max_model_len=1024
```

**Important:** `max_response_length` is **ignored** in continuous generation mode. The agent will continue generating until `max_model_len` is reached, budget is exhausted, or other termination conditions are met.

### Interval-Based Generation

Instead of generating all tokens at once, continuous generation produces tokens in intervals. Each interval triggers budget checking, allowing fine-grained control over generation.

```bash
actor_rollout_ref.rollout.budget_checker.interval=100
```

- Generation happens in chunks of `interval` tokens (e.g., 100 tokens at a time)
- After each interval, the budget checker is called
- Last interval is automatically calculated as `min(interval, remaining_tokens)`
- Budget checking every token would be slow; intervals provide a performance/precision tradeoff

**Trade-off:** The agent can exceed budget by up to `interval` tokens in the worst case (when budget is exhausted mid-interval).

## Configuration

### Enabling Continuous Generation

Continuous generation is activated through the tool agent loop:

```bash
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.rollout.multi_turn.format=hermes  # or other tool formats
```

### EOS Prevention

To keep generation going and prevent premature termination, you can suppress specific tokens (like EOS):

```bash
actor_rollout_ref.rollout.prevent_eos_generation=True
actor_rollout_ref.rollout.suppressed_token_ids='[151643,151645]'  # Token IDs to suppress
actor_rollout_ref.rollout.suppressed_tokens_logit_bias=-100.0     # Strong negative bias
```

**How it works:**
- `prevent_eos_generation=True` enables the feature
- `suppressed_token_ids` lists token IDs to suppress (e.g., EOS, specific special tokens)
- `suppressed_tokens_logit_bias` sets the logit bias for suppressed tokens (negative values reduce probability)

This prevents the model from generating suppressed tokens, keeping the rollout active.

**Finding which tokens to suppress:**

Check your model's tokenizer configuration files (usually `tokenizer_config.json` and `tokenizer.json` in the model directory) to identify special tokens that stop generation. Look for:
- `eos_token` and `eos_token_id`
- Chat-specific end markers (e.g., `<|im_end|>`, `<|endoftext|>`)
- Any tokens marking end of assistant turns

For example, Qwen3-0.6B uses token IDs `[151643,151645]` for its EOS and chat end tokens.


### Custom Stop Strings

You can specify custom stop strings to halt generation cleanly:

```bash
actor_rollout_ref.rollout.stop='["</tool_call>"]'
```

**Why use this:**
- Standard generation stops at EOS tokens, which may not align with your format
- Stop strings provide cleaner, more reliable termination points
- Useful for tool calling formats, structured outputs, or custom delimiters
- More flexible than relying solely on EOS tokens

**Example use cases:**
- `["</tool_call>"]` - Stop after tool call closing tag
- `["</budget_check">]` - Stop to append remaining budget into the context-window

### Budget Checker

Budget checking allows you to control generation based on custom logic (character count, token count, cost, or any other criteria).

#### Configuration

```bash
actor_rollout_ref.rollout.budget_checker.path=/root/verl/verl/workers/rollout/budget_checker.py
actor_rollout_ref.rollout.budget_checker.name=character_count_budget_checker
actor_rollout_ref.rollout.budget_checker.interval=100
```

- **`path`**: Path to Python file containing the budget checker function
- **`name`**: Function name to import and use
- **`interval`**: How often to check budget (in tokens)

#### Function Signature

Budget checker functions must follow this signature:

```python
def my_budget_checker(
    text: str,              # Full conversation (prompt + all generated tokens)
    token_ids: list[int],   # Full conversation token IDs
    total_tokens: int       # Number of tokens generated (response only, excluding prompt)
) -> bool:
    """
    Returns:
        True: Continue generation
        False: Stop generation (budget exhausted)
    """
    # Your logic here
    return should_continue
```

**Parameters explained:**
- `text`: Decoded text of the entire conversation (original prompt + all generated response)
- `token_ids`: Token IDs of the entire conversation
- `total_tokens`: Count of tokens generated in the response (excludes original prompt length)

**Return value:**
- `True`: Budget check passed, continue generating
- `False`: Budget exhausted, terminate generation

#### Built-in Budget Checkers

VERL provides two built-in budget checkers in `/verl/workers/rollout/budget_checker.py`:

**1. Character Count Budget Checker**
```python
def character_count_budget_checker(text: str, token_ids: list[int], total_tokens: int) -> bool:
    """Stops when text exceeds 500 characters."""
    return len(text) < 500
```

**2. Token Count Budget Checker**
```python
def token_count_budget_checker(text: str, token_ids: list[int], total_tokens: int) -> bool:
    """Stops when total tokens exceed 1000."""
    return total_tokens < 1000
```

#### Custom Budget Checker Example

Create your own budget checker for custom logic:

```python
# my_budget_checker.py
def cost_based_budget_checker(text: str, token_ids: list[int], total_tokens: int) -> bool:
    """Stop when estimated cost exceeds $0.10"""
    COST_PER_TOKEN = 0.0001  # Example pricing
    estimated_cost = total_tokens * COST_PER_TOKEN
    return estimated_cost < 0.10

def complexity_budget_checker(text: str, token_ids: list[int], total_tokens: int) -> bool:
    """Stop when conversation gets too complex (e.g., too many tool calls)"""
    tool_call_count = text.count("<tool_call>")
    return tool_call_count < 5

def combined_budget_checker(text: str, token_ids: list[int], total_tokens: int) -> bool:
    """Combine multiple constraints"""
    if total_tokens > 2000:
        return False
    if len(text) > 5000:
        return False
    if text.count("<tool_call>") > 10:
        return False
    return True
```

Then use it:
```bash
actor_rollout_ref.rollout.budget_checker.path=/path/to/my_budget_checker.py
actor_rollout_ref.rollout.budget_checker.name=combined_budget_checker
```

## State Machine Flow

The continuous generation agent loop follows this state machine:

```
PENDING → GENERATING → CHECKING_BUDGET → [Budget OK?]
            ↑                                   ↓
            │                           ┌───────┴────────┐
            │                           │                │
            │                      [Tool Call?]    [Budget Exhausted/
            │                           │          Max Model Len]
            │                    ┌──────┴──────┐         │
            │                    │             │         │
            │                   YES           NO         ↓
            │                    │             │    TERMINATED
            │                    ↓             │         (END)
            │            PROCESSING_TOOLS      │
            │            /INTERACTING          │
            │                    │             │
            └────────────────────┴─────────────┘
```

**Flow explanation:**

1. **PENDING**: Initialize prompt, prepare for first generation
2. **GENERATING**: Generate one interval of tokens (up to `interval` size)
3. **CHECKING_BUDGET**: Check budget and termination conditions
   - If **budget exhausted** or **max_model_len reached** → **TERMINATED**
   - If **budget OK** → Check for tool calls
     - **Tool call found** → **PROCESSING_TOOLS** (or **INTERACTING** if user interaction) → back to **GENERATING**
     - **No tool call** → back to **GENERATING** (continuous generation)

**The loop continues** (GENERATING → CHECKING_BUDGET → GENERATING) **until:**
- Budget checker returns `False` (budget exhausted)
- `len(prompt_ids) >= max_model_len` (context window full)
- Max turns reached (`assistant_turns >= max_assistant_turns` or `user_turns >= max_user_turns`)
- Tool response would overflow `max_model_len`
- User interaction requests termination

**Key insight:** Tool processing happens **within the continuous generation loop**. After tools execute and responses are added to context, generation immediately resumes until budget/context limits are hit.

## Complete Example Configuration

Here's a complete example using continuous generation with budget checking:

```bash
python3 -m verl.trainer.main_ppo \
    # ... other configs ...
    
    # Enable continuous generation via tool agent
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    
    # Context window limit (fundamental limit)
    actor_rollout_ref.rollout.max_model_len=2048 \
    
    # max_response_length is IGNORED in continuous generation
    data.max_response_length=1024 \
    
    # Prevent EOS from stopping generation
    actor_rollout_ref.rollout.prevent_eos_generation=True \
    actor_rollout_ref.rollout.suppressed_token_ids='[151643,151645]' \
    actor_rollout_ref.rollout.suppressed_tokens_logit_bias=-100.0 \
    
    # Custom stop string for clean termination
    actor_rollout_ref.rollout.stop='["</tool_call>"]' \
    
    # Budget checker configuration
    actor_rollout_ref.rollout.budget_checker.interval=100 \
    actor_rollout_ref.rollout.budget_checker.path=/root/verl/verl/workers/rollout/budget_checker.py \
    actor_rollout_ref.rollout.budget_checker.name=character_count_budget_checker \
```

## Known Limitations and Edge Cases

### Budget Overshoot
The agent can exceed budget by up to `interval` tokens in the worst case. This happens when the budget limit is reached mid-interval. To minimize overshoot, use a smaller interval (at the cost of more frequent checks and slower generation).

### Tool Response Termination
If adding a tool response would exceed `max_model_len`, the rollout terminates **before** adding the tool response. This means:
- The model makes a tool call but never sees the result
- The training data won't include tool response for these cases
- May affect learning of tool usage patterns

**Mitigation:** Set `max_model_len` large enough to accommodate tool responses, or reserve buffer space in your budget checker.

### Partial Last Intervals
The final interval is automatically calculated as `min(interval, remaining_tokens)`. This is expected behavior and ensures the agent fills the context window exactly to `max_model_len`.

Example: With `max_model_len=1024` and `interval=100`, if the prompt is 243 tokens, the agent will generate:
- Intervals 1-7: 100 tokens each (700 tokens)
- Interval 8: 81 tokens (fills to exactly 1024)

## Comparison: Continuous vs Standard Rollout

| Feature | Standard Rollout | Continuous Generation |
|---------|-----------------|----------------------|
| **Generation limit** | `max_response_length` | `max_model_len` |
| **Budget checking** | Not available | Configurable with intervals |
| **EOS handling** | Stops at EOS | Can suppress EOS tokens |
| **Stop strings** | Limited | Fully configurable |
| **Multi-turn** | Fixed turns | Dynamic until budget/context exhausted |
| **Use case** | Fixed-length responses | Agentic workflows, complex reasoning |

```