---
title: "Introduction"
source: "https://ai.pydantic.dev/"
author:
published:
created: 2025-02-11
description: "Agent Framework / shim to use Pydantic with LLMs"
tags:
  - "clippings"
---
Version Notice

This documentation is ahead of the last release by [3 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.23...main). You may see documentation for features not yet supported in the latest release [v0.0.23 2025-02-07](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.23).

*Agent Framework / shim to use Pydantic with LLMs*

 [![CI](https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml?query=branch%3Amain)[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/pydantic/pydantic-ai.svg) ](https://coverage-badge.samuelcolvin.workers.dev/redirect/pydantic/pydantic-ai)[![PyPI](https://img.shields.io/pypi/v/pydantic-ai.svg) ](https://pypi.python.org/pypi/pydantic-ai)[![versions](https://img.shields.io/pypi/pyversions/pydantic-ai.svg) ](https://github.com/pydantic/pydantic-ai)[![license](https://img.shields.io/github/license/pydantic/pydantic-ai.svg)](https://github.com/pydantic/pydantic-ai/blob/main/LICENSE)

PydanticAI is a Python agent framework designed to make it less painful to build production grade applications with Generative AI.

PydanticAI is a Python Agent Framework designed to make it less painful to build production grade applications with Generative AI.

FastAPI revolutionized web development by offering an innovative and ergonomic design, built on the foundation of [Pydantic](https://docs.pydantic.dev/).

Similarly, virtually every agent framework and LLM library in Python uses Pydantic, yet when we began to use LLMs in [Pydantic Logfire](https://pydantic.dev/logfire), we couldn't find anything that gave us the same feeling.

We built PydanticAI with one simple aim: to bring that FastAPI feeling to GenAI app development.

## Why use PydanticAI

- **Built by the Pydantic Team**: Built by the team behind [Pydantic](https://docs.pydantic.dev/latest/) (the validation layer of the OpenAI SDK, the Anthropic SDK, LangChain, LlamaIndex, AutoGPT, Transformers, CrewAI, Instructor and many more).
- **Model-agnostic**: Supports OpenAI, Anthropic, Gemini, Deepseek, Ollama, Groq, Cohere, and Mistral, and there is a simple interface to implement support for [other models](https://ai.pydantic.dev/models/).
- **Pydantic Logfire Integration**: Seamlessly [integrates](https://ai.pydantic.dev/logfire/) with [Pydantic Logfire](https://pydantic.dev/logfire) for real-time debugging, performance monitoring, and behavior tracking of your LLM-powered applications.
- **Type-safe**: Designed to make [type checking](https://ai.pydantic.dev/agents/#static-type-checking) as powerful and informative as possible for you.
- **Python-centric Design**: Leverages Python's familiar control flow and agent composition to build your AI-driven projects, making it easy to apply standard Python best practices you'd use in any other (non-AI) project.
- **Structured Responses**: Harnesses the power of [Pydantic](https://docs.pydantic.dev/latest/) to [validate and structure](https://ai.pydantic.dev/results/#structured-result-validation) model outputs, ensuring responses are consistent across runs.
- **Dependency Injection System**: Offers an optional [dependency injection](https://ai.pydantic.dev/dependencies/) system to provide data and services to your agent's [system prompts](https://ai.pydantic.dev/agents/#system-prompts), [tools](https://ai.pydantic.dev/tools/) and [result validators](https://ai.pydantic.dev/results/#result-validators-functions). This is useful for testing and eval-driven iterative development.
- **Streamed Responses**: Provides the ability to [stream](https://ai.pydantic.dev/results/#streamed-results) LLM outputs continuously, with immediate validation, ensuring rapid and accurate results.
- **Graph Support**: [Pydantic Graph](https://ai.pydantic.dev/graph/) provides a powerful way to define graphs using typing hints, this is useful in complex applications where standard control flow can degrade to spaghetti code.

In Beta

PydanticAI is in early beta, the API is still subject to change and there's a lot more to do. [Feedback](https://github.com/pydantic/pydantic-ai/issues) is very welcome!

## Hello World Example

Here's a minimal example of PydanticAI:

hello\_world.py

```python
from pydantic_ai import Agent

agent = Agent(  
    'google-gla:gemini-1.5-flash',
    system_prompt='Be concise, reply with one sentence.',  
)

result = agent.run_sync('Where does "hello world" come from?')  
print(result.data)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""
```

*(This example is complete, it can be run "as is")*

The exchange should be very short: PydanticAI will send the system prompt and the user query to the LLM, the model will return a text response.

Not very interesting yet, but we can easily add "tools", dynamic system prompts, and structured responses to build more powerful agents.

Here is a concise example using PydanticAI to build a support agent for a bank:

bank\_support.py

```python
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from bank_database import DatabaseConn

@dataclass
class SupportDependencies:  
    customer_id: int
    db: DatabaseConn  

class SupportResult(BaseModel):  
    support_advice: str = Field(description='Advice returned to the customer')
    block_card: bool = Field(description="Whether to block the customer's card")
    risk: int = Field(description='Risk level of query', ge=0, le=10)

support_agent = Agent(  
    'openai:gpt-4o',  
    deps_type=SupportDependencies,
    result_type=SupportResult,  
    system_prompt=(  
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query.'
    ),
)

@support_agent.system_prompt  
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"

@support_agent.tool  
async def customer_balance(
    ctx: RunContext[SupportDependencies], include_pending: bool
) -> float:
    """Returns the customer's current account balance."""  
    return await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )

...  

async def main():
    deps = SupportDependencies(customer_id=123, db=DatabaseConn())
    result = await support_agent.run('What is my balance?', deps=deps)  
    print(result.data)  
    """
    support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
    """

    result = await support_agent.run('I just lost my card!', deps=deps)
    print(result.data)
    """
    support_advice="I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions." block_card=True risk=8
    """
```

Complete `bank_support.py` example

The code included here is incomplete for the sake of brevity (the definition of `DatabaseConn` is missing); you can find the complete `bank_support.py` example [here](https://ai.pydantic.dev/examples/bank-support/).

## Instrumentation with Pydantic Logfire

To understand the flow of the above runs, we can watch the agent in action using Pydantic Logfire.

To do this, we need to set up logfire, and add the following to our code:

bank\_support\_with\_logfire.py

```python
...
from bank_database import DatabaseConn

import logfire
logfire.configure()  
logfire.instrument_asyncpg()  
...
```

That's enough to get the following view of your agent in action:

See [Monitoring and Performance](https://ai.pydantic.dev/logfire/) to learn more.

## Next Steps

To try PydanticAI yourself, follow the instructions [in the examples](https://ai.pydantic.dev/examples/).

Read the [docs](https://ai.pydantic.dev/agents/) to learn more about building applications with PydanticAI.

Read the [API Reference](https://ai.pydantic.dev/api/agent/) to understand PydanticAI's interface.
---
title: "Installation - PydanticAI"
source: "https://ai.pydantic.dev/install/"
author:
published:
created: 2025-02-11
description: "Agent Framework / shim to use Pydantic with LLMs"
tags:
  - "clippings"
---
Version Notice

This documentation is ahead of the last release by [3 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.23...main). You may see documentation for features not yet supported in the latest release [v0.0.23 2025-02-07](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.23).

PydanticAI is available on PyPI as [`pydantic-ai`](https://pypi.org/project/pydantic-ai/) so installation is as simple as:

(Requires Python 3.9+)

This installs the `pydantic_ai` package, core dependencies, and libraries required to use all the models included in PydanticAI. If you want to use a specific model, you can install the ["slim"](https://ai.pydantic.dev/install/#slim-install) version of PydanticAI.

## Use with Pydantic Logfire

PydanticAI has an excellent (but completely optional) integration with [Pydantic Logfire](https://pydantic.dev/logfire) to help you view and understand agent runs.

To use Logfire with PydanticAI, install `pydantic-ai` or `pydantic-ai-slim` with the `logfire` optional group:

```
pip install 'pydantic-ai[logfire]'
```

```
uv add 'pydantic-ai[logfire]'
```

From there, follow the [Logfire setup docs](https://ai.pydantic.dev/logfire/#using-logfire) to configure Logfire.

## Running Examples

We distribute the [`pydantic_ai_examples`](https://github.com/pydantic/pydantic-ai/tree/main/examples/pydantic_ai_examples) directory as a separate PyPI package ([`pydantic-ai-examples`](https://pypi.org/project/pydantic-ai-examples/)) to make examples extremely easy to customize and run.

To install examples, use the `examples` optional group:

```
pip install 'pydantic-ai[examples]'
```

```
uv add 'pydantic-ai[examples]'
```

To run the examples, follow instructions in the [examples docs](https://ai.pydantic.dev/examples/).

## Slim Install

If you know which model you're going to use and want to avoid installing superfluous packages, you can use the [`pydantic-ai-slim`](https://pypi.org/project/pydantic-ai-slim/) package. For example, if you're using just [`OpenAIModel`](https://ai.pydantic.dev/api/models/openai/#pydantic_ai.models.openai.OpenAIModel), you would run:

```
pip install 'pydantic-ai-slim[openai]'
```

```
uv add 'pydantic-ai-slim[openai]'
```

`pydantic-ai-slim` has the following optional groups:

- `logfire` — installs [`logfire`](https://ai.pydantic.dev/logfire/) [PyPI ↗](https://pypi.org/project/logfire)
- `openai` — installs `openai` [PyPI ↗](https://pypi.org/project/openai)
- `vertexai` — installs `google-auth` [PyPI ↗](https://pypi.org/project/google-auth) and `requests` [PyPI ↗](https://pypi.org/project/requests)
- `anthropic` — installs `anthropic` [PyPI ↗](https://pypi.org/project/anthropic)
- `groq` — installs `groq` [PyPI ↗](https://pypi.org/project/groq)
- `mistral` — installs `mistralai` [PyPI ↗](https://pypi.org/project/mistralai)
- `cohere` - installs `cohere` [PyPI ↗](https://pypi.org/project/cohere)

See the [models](https://ai.pydantic.dev/models/) documentation for information on which optional dependencies are required for each model.

You can also install dependencies for multiple models and use cases, for example:

```
pip install 'pydantic-ai-slim[openai,vertexai,logfire]'
```

```
uv add 'pydantic-ai-slim[openai,vertexai,logfire]'
```

---
title: "Troubleshooting - PydanticAI"
source: "https://ai.pydantic.dev/troubleshooting/"
author:
published:
created: 2025-02-11
description: "Agent Framework / shim to use Pydantic with LLMs"
tags:
  - "clippings"
---
Version Notice

This documentation is ahead of the last release by [3 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.23...main). You may see documentation for features not yet supported in the latest release [v0.0.23 2025-02-07](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.23).

Below are suggestions on how to fix some common errors you might encounter while using PydanticAI. If the issue you're experiencing is not listed below or addressed in the documentation, please feel free to ask in the [Pydantic Slack](https://ai.pydantic.dev/help/) or create an issue on [GitHub](https://github.com/pydantic/pydantic-ai/issues).

## Jupyter Notebook Errors

### `RuntimeError: This event loop is already running`

This error is caused by conflicts between the event loops in Jupyter notebook and PydanticAI's. One way to manage these conflicts is by using [`nest-asyncio`](https://pypi.org/project/nest-asyncio/). Namely, before you execute any agent runs, do the following:

```python
import nest_asyncio

nest_asyncio.apply()
```

Note: This fix also applies to Google Colab.

## API Key Configuration

### `UserError: API key must be provided or set in the [MODEL]_API_KEY environment variable`

If you're running into issues with setting the API key for your model, visit the [Models](https://ai.pydantic.dev/models/) page to learn more about how to set an environment variable and/or pass in an `api_key` argument.

## Monitoring HTTPX Requests

You can use custom `httpx` clients in your models in order to access specific requests, responses, and headers at runtime.

It's particularly helpful to use `logfire`'s [HTTPX integration](https://ai.pydantic.dev/logfire/#monitoring-httpx-requests) to monitor the above.

---
title: "Agents - PydanticAI"
source: "https://ai.pydantic.dev/agents/"
author:
published:
created: 2025-02-11
description: "Agent Framework / shim to use Pydantic with LLMs"
tags:
  - "clippings"
---
Version Notice

This documentation is ahead of the last release by [3 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.23...main). You may see documentation for features not yet supported in the latest release [v0.0.23 2025-02-07](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.23).

## Introduction

Agents are PydanticAI's primary interface for interacting with LLMs.

In some use cases a single Agent will control an entire application or component, but multiple agents can also interact to embody more complex workflows.

The [`Agent`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent) class has full API documentation, but conceptually you can think of an agent as a container for:

In typing terms, agents are generic in their dependency and result types, e.g., an agent which required dependencies of type `Foobar` and returned results of type `list[str]` would have type `Agent[Foobar, list[str]]`. In practice, you shouldn't need to care about this, it should just mean your IDE can tell you when you have the right type, and if you choose to use [static type checking](https://ai.pydantic.dev/agents/#static-type-checking) it should work well with PydanticAI.

Here's a toy example of an agent that simulates a roulette wheel:

roulette\_wheel.py

```python
from pydantic_ai import Agent, RunContext

roulette_agent = Agent(  
    'openai:gpt-4o',
    deps_type=int,
    result_type=bool,
    system_prompt=(
        'Use the \`roulette_wheel\` function to see if the '
        'customer has won based on the number they provide.'
    ),
)

@roulette_agent.tool
async def roulette_wheel(ctx: RunContext[int], square: int) -> str:  
    """check if the square is a winner"""
    return 'winner' if square == ctx.deps else 'loser'

# Run the agent
success_number = 18  
result = roulette_agent.run_sync('Put my money on square eighteen', deps=success_number)
print(result.data)  
#> True

result = roulette_agent.run_sync('I bet five is the winner', deps=success_number)
print(result.data)
#> False
```

Agents are designed for reuse, like FastAPI Apps

Agents are intended to be instantiated once (frequently as module globals) and reused throughout your application, similar to a small [FastAPI](https://fastapi.tiangolo.com/reference/fastapi/#fastapi.FastAPI) app or an [APIRouter](https://fastapi.tiangolo.com/reference/apirouter/#fastapi.APIRouter).

## Running Agents

There are three ways to run an agent:

1. [`agent.run()`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run) — a coroutine which returns a [`RunResult`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.RunResult) containing a completed response
2. [`agent.run_sync()`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run_sync) — a plain, synchronous function which returns a [`RunResult`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.RunResult) containing a completed response (internally, this just calls `loop.run_until_complete(self.run())`)
3. [`agent.run_stream()`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run_stream) — a coroutine which returns a [`StreamedRunResult`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.StreamedRunResult), which contains methods to stream a response as an async iterable

Here's a simple example demonstrating all three:

run\_agent.py

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

result_sync = agent.run_sync('What is the capital of Italy?')
print(result_sync.data)
#> Rome

async def main():
    result = await agent.run('What is the capital of France?')
    print(result.data)
    #> Paris

    async with agent.run_stream('What is the capital of the UK?') as response:
        print(await response.get_data())
        #> London
```

*(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)*

You can also pass messages from previous runs to continue a conversation or provide context, as described in [Messages and Chat History](https://ai.pydantic.dev/message-history/).

### Additional Configuration

#### Usage Limits

PydanticAI offers a [`UsageLimits`](https://ai.pydantic.dev/api/usage/#pydantic_ai.usage.UsageLimits) structure to help you limit your usage (tokens and/or requests) on model runs.

You can apply these settings by passing the `usage_limits` argument to the `run{_sync,_stream}` functions.

Consider the following example, where we limit the number of response tokens:

```py
from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

agent = Agent('anthropic:claude-3-5-sonnet-latest')

result_sync = agent.run_sync(
    'What is the capital of Italy? Answer with just the city.',
    usage_limits=UsageLimits(response_tokens_limit=10),
)
print(result_sync.data)
#> Rome
print(result_sync.usage())
"""
Usage(requests=1, request_tokens=62, response_tokens=1, total_tokens=63, details=None)
"""

try:
    result_sync = agent.run_sync(
        'What is the capital of Italy? Answer with a paragraph.',
        usage_limits=UsageLimits(response_tokens_limit=10),
    )
except UsageLimitExceeded as e:
    print(e)
    #> Exceeded the response_tokens_limit of 10 (response_tokens=32)
```

Restricting the number of requests can be useful in preventing infinite loops or excessive tool calling:

```py
from typing_extensions import TypedDict

from pydantic_ai import Agent, ModelRetry
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

class NeverResultType(TypedDict):
    """
    Never ever coerce data to this type.
    """

    never_use_this: str

agent = Agent(
    'anthropic:claude-3-5-sonnet-latest',
    retries=3,
    result_type=NeverResultType,
    system_prompt='Any time you get a response, call the \`infinite_retry_tool\` to produce another response.',
)

@agent.tool_plain(retries=5)  
def infinite_retry_tool() -> int:
    raise ModelRetry('Please try again.')

try:
    result_sync = agent.run_sync(
        'Begin infinite retry loop!', usage_limits=UsageLimits(request_limit=3)  
    )
except UsageLimitExceeded as e:
    print(e)
    #> The next request would exceed the request_limit of 3
```

Note

This is especially relevant if you're registered a lot of tools, `request_limit` can be used to prevent the model from choosing to make too many of these calls.

#### Model (Run) Settings

PydanticAI offers a [`settings.ModelSettings`](https://ai.pydantic.dev/api/settings/#pydantic_ai.settings.ModelSettings) structure to help you fine tune your requests. This structure allows you to configure common parameters that influence the model's behavior, such as `temperature`, `max_tokens`, `timeout`, and more.

There are two ways to apply these settings: 1. Passing to `run{_sync,_stream}` functions via the `model_settings` argument. This allows for fine-tuning on a per-request basis. 2. Setting during [`Agent`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent) initialization via the `model_settings` argument. These settings will be applied by default to all subsequent run calls using said agent. However, `model_settings` provided during a specific run call will override the agent's default settings.

For example, if you'd like to set the `temperature` setting to `0.0` to ensure less random behavior, you can do the following:

```py
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

result_sync = agent.run_sync(
    'What is the capital of Italy?', model_settings={'temperature': 0.0}
)
print(result_sync.data)
#> Rome
```

### Model specific settings

If you wish to further customize model behavior, you can use a subclass of [`ModelSettings`](https://ai.pydantic.dev/api/settings/#pydantic_ai.settings.ModelSettings), like [`GeminiModelSettings`](https://ai.pydantic.dev/api/models/gemini/#pydantic_ai.models.gemini.GeminiModelSettings), associated with your model of choice.

For example:

```py
from pydantic_ai import Agent, UnexpectedModelBehavior
from pydantic_ai.models.gemini import GeminiModelSettings

agent = Agent('google-gla:gemini-1.5-flash')

try:
    result = agent.run_sync(
        'Write a list of 5 very rude things that I might say to the universe after stubbing my toe in the dark:',
        model_settings=GeminiModelSettings(
            temperature=0.0,  # general model settings can also be specified
            gemini_safety_settings=[
                {
                    'category': 'HARM_CATEGORY_HARASSMENT',
                    'threshold': 'BLOCK_LOW_AND_ABOVE',
                },
                {
                    'category': 'HARM_CATEGORY_HATE_SPEECH',
                    'threshold': 'BLOCK_LOW_AND_ABOVE',
                },
            ],
        ),
    )
except UnexpectedModelBehavior as e:
    print(e)  
    """
    Safety settings triggered, body:
    <safety settings details>
    """
```

## Runs vs. Conversations

An agent **run** might represent an entire conversation — there's no limit to how many messages can be exchanged in a single run. However, a **conversation** might also be composed of multiple runs, especially if you need to maintain state between separate interactions or API calls.

Here's an example of a conversation comprised of multiple runs:

conversation\_example.py

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

# First run
result1 = agent.run_sync('Who was Albert Einstein?')
print(result1.data)
#> Albert Einstein was a German-born theoretical physicist.

# Second run, passing previous messages
result2 = agent.run_sync(
    'What was his most famous equation?',
    message_history=result1.new_messages(),  
)
print(result2.data)
#> Albert Einstein's most famous equation is (E = mc^2).
```

*(This example is complete, it can be run "as is")*

## Type safe by design

PydanticAI is designed to work well with static type checkers, like mypy and pyright.

Typing is (somewhat) optional

PydanticAI is designed to make type checking as useful as possible for you if you choose to use it, but you don't have to use types everywhere all the time.

That said, because PydanticAI uses Pydantic, and Pydantic uses type hints as the definition for schema and validation, some types (specifically type hints on parameters to tools, and the `result_type` arguments to [`Agent`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent)) are used at runtime.

We (the library developers) have messed up if type hints are confusing you more than helping you, if you find this, please create an [issue](https://github.com/pydantic/pydantic-ai/issues) explaining what's annoying you!

In particular, agents are generic in both the type of their dependencies and the type of results they return, so you can use the type hints to ensure you're using the right types.

Consider the following script with type mistakes:

type\_mistakes.py

```python
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

@dataclass
class User:
    name: str

agent = Agent(
    'test',
    deps_type=User,  
    result_type=bool,
)

@agent.system_prompt
def add_user_name(ctx: RunContext[str]) -> str:  
    return f"The user's name is {ctx.deps}."

def foobar(x: bytes) -> None:
    pass

result = agent.run_sync('Does their name start with "A"?', deps=User('Anne'))
foobar(result.data)  
```

Running `mypy` on this will give the following output:

```bash
➤ uv run mypy type_mistakes.py
type_mistakes.py:18: error: Argument 1 to "system_prompt" of "Agent" has incompatible type "Callable[[RunContext[str]], str]"; expected "Callable[[RunContext[User]], str]"  [arg-type]
type_mistakes.py:28: error: Argument 1 to "foobar" has incompatible type "bool"; expected "bytes"  [arg-type]
Found 2 errors in 1 file (checked 1 source file)
```

Running `pyright` would identify the same issues.

## System Prompts

System prompts might seem simple at first glance since they're just strings (or sequences of strings that are concatenated), but crafting the right system prompt is key to getting the model to behave as you want.

Generally, system prompts fall into two categories:

1. **Static system prompts**: These are known when writing the code and can be defined via the `system_prompt` parameter of the [`Agent` constructor](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.__init__).
2. **Dynamic system prompts**: These depend in some way on context that isn't known until runtime, and should be defined via functions decorated with [`@agent.system_prompt`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.system_prompt).

You can add both to a single agent; they're appended in the order they're defined at runtime.

Here's an example using both types of system prompts:

system\_prompts.py

```python
from datetime import date

from pydantic_ai import Agent, RunContext

agent = Agent(
    'openai:gpt-4o',
    deps_type=str,  
    system_prompt="Use the customer's name while replying to them.",  
)

@agent.system_prompt  
def add_the_users_name(ctx: RunContext[str]) -> str:
    return f"The user's name is {ctx.deps}."

@agent.system_prompt
def add_the_date() -> str:  
    return f'The date is {date.today()}.'

result = agent.run_sync('What is the date?', deps='Frank')
print(result.data)
#> Hello Frank, the date today is 2032-01-02.
```

*(This example is complete, it can be run "as is")*

## Reflection and self-correction

Validation errors from both function tool parameter validation and [structured result validation](https://ai.pydantic.dev/results/#structured-result-validation) can be passed back to the model with a request to retry.

You can also raise [`ModelRetry`](https://ai.pydantic.dev/api/exceptions/#pydantic_ai.exceptions.ModelRetry) from within a [tool](https://ai.pydantic.dev/tools/) or [result validator function](https://ai.pydantic.dev/results/#result-validators-functions) to tell the model it should retry generating a response.

- The default retry count is **1** but can be altered for the [entire agent](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.__init__), a [specific tool](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.tool), or a [result validator](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.__init__).
- You can access the current retry count from within a tool or result validator via [`ctx.retry`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext).

Here's an example:

tool\_retry.py

```python
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext, ModelRetry

from fake_database import DatabaseConn

class ChatResult(BaseModel):
    user_id: int
    message: str

agent = Agent(
    'openai:gpt-4o',
    deps_type=DatabaseConn,
    result_type=ChatResult,
)

@agent.tool(retries=2)
def get_user_by_name(ctx: RunContext[DatabaseConn], name: str) -> int:
    """Get a user's ID from their full name."""
    print(name)
    #> John
    #> John Doe
    user_id = ctx.deps.users.get(name=name)
    if user_id is None:
        raise ModelRetry(
            f'No user found with name {name!r}, remember to provide their full name'
        )
    return user_id

result = agent.run_sync(
    'Send a message to John Doe asking for coffee next week', deps=DatabaseConn()
)
print(result.data)
"""
user_id=123 message='Hello John, would you be free for coffee sometime next week? Let me know what works for you!'
"""
```

## Model errors

If models behave unexpectedly (e.g., the retry limit is exceeded, or their API returns `503`), agent runs will raise [`UnexpectedModelBehavior`](https://ai.pydantic.dev/api/exceptions/#pydantic_ai.exceptions.UnexpectedModelBehavior).

In these cases, [`capture_run_messages`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.capture_run_messages) can be used to access the messages exchanged during the run to help diagnose the issue.

```python
from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior, capture_run_messages

agent = Agent('openai:gpt-4o')

@agent.tool_plain
def calc_volume(size: int) -> int:  
    if size == 42:
        return size**3
    else:
        raise ModelRetry('Please try again.')

with capture_run_messages() as messages:  
    try:
        result = agent.run_sync('Please get me the volume of a box with size 6.')
    except UnexpectedModelBehavior as e:
        print('An error occurred:', e)
        #> An error occurred: Tool exceeded max retries count of 1
        print('cause:', repr(e.__cause__))
        #> cause: ModelRetry('Please try again.')
        print('messages:', messages)
        """
        messages:
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Please get me the volume of a box with size 6.',
                        timestamp=datetime.datetime(...),
                        part_kind='user-prompt',
                    )
                ],
                kind='request',
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='calc_volume',
                        args={'size': 6},
                        tool_call_id=None,
                        part_kind='tool-call',
                    )
                ],
                model_name='function:model_logic',
                timestamp=datetime.datetime(...),
                kind='response',
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Please try again.',
                        tool_name='calc_volume',
                        tool_call_id=None,
                        timestamp=datetime.datetime(...),
                        part_kind='retry-prompt',
                    )
                ],
                kind='request',
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='calc_volume',
                        args={'size': 6},
                        tool_call_id=None,
                        part_kind='tool-call',
                    )
                ],
                model_name='function:model_logic',
                timestamp=datetime.datetime(...),
                kind='response',
            ),
        ]
        """
    else:
        print(result.data)
```

*(This example is complete, it can be run "as is")*

Note

If you call [`run`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run), [`run_sync`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run_sync), or [`run_stream`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run_stream) more than once within a single `capture_run_messages` context, `messages` will represent the messages exchanged during the first call only.

---
title: "Models - PydanticAI"
source: "https://ai.pydantic.dev/models/#environment-variable"
author:
published:
created: 2025-02-11
description: "Agent Framework / shim to use Pydantic with LLMs"
tags:
  - "clippings"
---
Version Notice

This documentation is ahead of the last release by [3 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.23...main). You may see documentation for features not yet supported in the latest release [v0.0.23 2025-02-07](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.23).

PydanticAI is Model-agnostic and has built in support for the following model providers:

- [OpenAI](https://ai.pydantic.dev/models/#openai)
- [Anthropic](https://ai.pydantic.dev/models/#anthropic)
- Gemini via two different APIs: [Generative Language API](https://ai.pydantic.dev/models/#gemini) and [VertexAI API](https://ai.pydantic.dev/models/#gemini-via-vertexai)
- [Ollama](https://ai.pydantic.dev/models/#ollama)
- [Deepseek](https://ai.pydantic.dev/models/#deepseek)
- [Groq](https://ai.pydantic.dev/models/#groq)
- [Mistral](https://ai.pydantic.dev/models/#mistral)
- [Cohere](https://ai.pydantic.dev/models/#cohere)

See [OpenAI-compatible models](https://ai.pydantic.dev/models/#openai-compatible-models) for more examples on how to use models such as [OpenRouter](https://ai.pydantic.dev/models/#openrouter), and [Grok (xAI)](https://ai.pydantic.dev/models/#grok-xai) that support the OpenAI SDK.

You can also [add support for other models](https://ai.pydantic.dev/models/#implementing-custom-models).

PydanticAI also comes with [`TestModel`](https://ai.pydantic.dev/api/models/test/) and [`FunctionModel`](https://ai.pydantic.dev/api/models/function/) for testing and development.

To use each model provider, you need to configure your local environment and make sure you have the right packages installed.

## OpenAI

### Install

To use OpenAI models, you need to either install [`pydantic-ai`](https://ai.pydantic.dev/install/), or install [`pydantic-ai-slim`](https://ai.pydantic.dev/install/#slim-install) with the `openai` optional group:

```
pip install 'pydantic-ai-slim[openai]'
```

```
uv add 'pydantic-ai-slim[openai]'
```

### Configuration

To use [`OpenAIModel`](https://ai.pydantic.dev/api/models/openai/#pydantic_ai.models.openai.OpenAIModel) through their main API, go to [platform.openai.com](https://platform.openai.com/) and follow your nose until you find the place to generate an API key.

### Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key'
```

You can then use [`OpenAIModel`](https://ai.pydantic.dev/api/models/openai/#pydantic_ai.models.openai.OpenAIModel) by name:

openai\_model\_by\_name.py

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')
...
```

Or initialise the model directly with just the model name:

openai\_model\_init.py

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

model = OpenAIModel('gpt-4o')
agent = Agent(model)
...
```

### `api_key` argument

If you don't want to or can't set the environment variable, you can pass it at runtime via the [`api_key` argument](https://ai.pydantic.dev/api/models/openai/#pydantic_ai.models.openai.OpenAIModel.__init__):

openai\_model\_api\_key.py

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

model = OpenAIModel('gpt-4o', api_key='your-api-key')
agent = Agent(model)
...
```

### Custom OpenAI Client

`OpenAIModel` also accepts a custom `AsyncOpenAI` client via the [`openai_client` parameter](https://ai.pydantic.dev/api/models/openai/#pydantic_ai.models.openai.OpenAIModel.__init__), so you can customise the `organization`, `project`, `base_url` etc. as defined in the [OpenAI API docs](https://platform.openai.com/docs/api-reference).

You could also use the [`AsyncAzureOpenAI`](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/switching-endpoints) client to use the Azure OpenAI API.

openai\_azure.py

```python
from openai import AsyncAzureOpenAI

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

client = AsyncAzureOpenAI(
    azure_endpoint='...',
    api_version='2024-07-01-preview',
    api_key='your-api-key',
)

model = OpenAIModel('gpt-4o', openai_client=client)
agent = Agent(model)
...
```

## Anthropic

### Install

To use [`AnthropicModel`](https://ai.pydantic.dev/api/models/anthropic/#pydantic_ai.models.anthropic.AnthropicModel) models, you need to either install [`pydantic-ai`](https://ai.pydantic.dev/install/), or install [`pydantic-ai-slim`](https://ai.pydantic.dev/install/#slim-install) with the `anthropic` optional group:

```
pip install 'pydantic-ai-slim[anthropic]'
```

```
uv add 'pydantic-ai-slim[anthropic]'
```

### Configuration

To use [Anthropic](https://anthropic.com/) through their API, go to [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) to generate an API key.

[`AnthropicModelName`](https://ai.pydantic.dev/api/models/anthropic/#pydantic_ai.models.anthropic.AnthropicModelName) contains a list of available Anthropic models.

### Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export ANTHROPIC_API_KEY='your-api-key'
```

You can then use [`AnthropicModel`](https://ai.pydantic.dev/api/models/anthropic/#pydantic_ai.models.anthropic.AnthropicModel) by name:

anthropic\_model\_by\_name.py

```py
from pydantic_ai import Agent

agent = Agent('anthropic:claude-3-5-sonnet-latest')
...
```

Or initialise the model directly with just the model name:

anthropic\_model\_init.py

```py
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel

model = AnthropicModel('claude-3-5-sonnet-latest')
agent = Agent(model)
...
```

### `api_key` argument

If you don't want to or can't set the environment variable, you can pass it at runtime via the [`api_key` argument](https://ai.pydantic.dev/api/models/anthropic/#pydantic_ai.models.anthropic.AnthropicModel.__init__):

anthropic\_model\_api\_key.py

```py
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel

model = AnthropicModel('claude-3-5-sonnet-latest', api_key='your-api-key')
agent = Agent(model)
...
```

## Gemini

For prototyping only

Google themselves refer to this API as the "hobby" API, I've received 503 responses from it a number of times. The API is easy to use and useful for prototyping and simple demos, but I would not rely on it in production.

If you want to run Gemini models in production, you should use the [VertexAI API](https://ai.pydantic.dev/models/#gemini-via-vertexai) described below.

### Install

To use [`GeminiModel`](https://ai.pydantic.dev/api/models/gemini/#pydantic_ai.models.gemini.GeminiModel) models, you just need to install [`pydantic-ai`](https://ai.pydantic.dev/install/) or [`pydantic-ai-slim`](https://ai.pydantic.dev/install/#slim-install), no extra dependencies are required.

### Configuration

[`GeminiModel`](https://ai.pydantic.dev/api/models/gemini/#pydantic_ai.models.gemini.GeminiModel) let's you use the Google's Gemini models through their [Generative Language API](https://ai.google.dev/api/all-methods), `generativelanguage.googleapis.com`.

[`GeminiModelName`](https://ai.pydantic.dev/api/models/gemini/#pydantic_ai.models.gemini.GeminiModelName) contains a list of available Gemini models that can be used through this interface.

To use `GeminiModel`, go to [aistudio.google.com](https://aistudio.google.com/) and follow your nose until you find the place to generate an API key.

### Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export GEMINI_API_KEY=your-api-key
```

You can then use [`GeminiModel`](https://ai.pydantic.dev/api/models/gemini/#pydantic_ai.models.gemini.GeminiModel) by name:

gemini\_model\_by\_name.py

```python
from pydantic_ai import Agent

agent = Agent('google-gla:gemini-1.5-flash')
...
```

Or initialise the model directly with just the model name:

gemini\_model\_init.py

```python
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

model = GeminiModel('gemini-1.5-flash')
agent = Agent(model)
...
```

### `api_key` argument

If you don't want to or can't set the environment variable, you can pass it at runtime via the [`api_key` argument](https://ai.pydantic.dev/api/models/gemini/#pydantic_ai.models.gemini.GeminiModel.__init__):

gemini\_model\_api\_key.py

```python
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

model = GeminiModel('gemini-1.5-flash', api_key='your-api-key')
agent = Agent(model)
...
```

## Gemini via VertexAI

To run Google's Gemini models in production, you should use [`VertexAIModel`](https://ai.pydantic.dev/api/models/vertexai/#pydantic_ai.models.vertexai.VertexAIModel) which uses the `*-aiplatform.googleapis.com` API.

[`GeminiModelName`](https://ai.pydantic.dev/api/models/gemini/#pydantic_ai.models.gemini.GeminiModelName) contains a list of available Gemini models that can be used through this interface.

### Install

To use [`VertexAIModel`](https://ai.pydantic.dev/api/models/vertexai/#pydantic_ai.models.vertexai.VertexAIModel), you need to either install [`pydantic-ai`](https://ai.pydantic.dev/install/), or install [`pydantic-ai-slim`](https://ai.pydantic.dev/install/#slim-install) with the `vertexai` optional group:

```
pip install 'pydantic-ai-slim[vertexai]'
```

```
uv add 'pydantic-ai-slim[vertexai]'
```

### Configuration

This interface has a number of advantages over `generativelanguage.googleapis.com` documented above:

1. The VertexAI API is more reliably and marginally lower latency in our experience.
2. You can [purchase provisioned throughput](https://cloud.google.com/vertex-ai/generative-ai/docs/provisioned-throughput#purchase-provisioned-throughput) with VertexAI to guarantee capacity.
3. If you're running PydanticAI inside GCP, you don't need to set up authentication, it should "just work".
4. You can decide which region to use, which might be important from a regulatory perspective, and might improve latency.

The big disadvantage is that for local development you may need to create and configure a "service account", which I've found extremely painful to get right in the past.

Whichever way you authenticate, you'll need to have VertexAI enabled in your GCP account.

### Application default credentials

Luckily if you're running PydanticAI inside GCP, or you have the [`gcloud` CLI](https://cloud.google.com/sdk/gcloud) installed and configured, you should be able to use `VertexAIModel` without any additional setup.

To use `VertexAIModel`, with [application default credentials](https://cloud.google.com/docs/authentication/application-default-credentials) configured (e.g. with `gcloud`), you can simply use:

vertexai\_application\_default\_credentials.py

```python
from pydantic_ai import Agent
from pydantic_ai.models.vertexai import VertexAIModel

model = VertexAIModel('gemini-1.5-flash')
agent = Agent(model)
...
```

Internally this uses [`google.auth.default()`](https://google-auth.readthedocs.io/en/master/reference/google.auth.html) from the `google-auth` package to obtain credentials.

Won't fail until `agent.run()`

Because `google.auth.default()` requires network requests and can be slow, it's not run until you call `agent.run()`. Meaning any configuration or permissions error will only be raised when you try to use the model. To initialize the model for this check to be run, call [`await model.ainit()`](https://ai.pydantic.dev/api/models/vertexai/#pydantic_ai.models.vertexai.VertexAIModel.ainit).

You may also need to pass the [`project_id` argument to `VertexAIModel`](https://ai.pydantic.dev/api/models/vertexai/#pydantic_ai.models.vertexai.VertexAIModel.__init__) if application default credentials don't set a project, if you pass `project_id` and it conflicts with the project set by application default credentials, an error is raised.

### Service account

If instead of application default credentials, you want to authenticate with a service account, you'll need to create a service account, add it to your GCP project (note: AFAIK this step is necessary even if you created the service account within the project), give that service account the "Vertex AI Service Agent" role, and download the service account JSON file.

Once you have the JSON file, you can use it thus:

vertexai\_service\_account.py

```python
from pydantic_ai import Agent
from pydantic_ai.models.vertexai import VertexAIModel

model = VertexAIModel(
    'gemini-1.5-flash',
    service_account_file='path/to/service-account.json',
)
agent = Agent(model)
...
```

### Customising region

Whichever way you authenticate, you can specify which region requests will be sent to via the [`region` argument](https://ai.pydantic.dev/api/models/vertexai/#pydantic_ai.models.vertexai.VertexAIModel.__init__).

Using a region close to your application can improve latency and might be important from a regulatory perspective.

vertexai\_region.py

```python
from pydantic_ai import Agent
from pydantic_ai.models.vertexai import VertexAIModel

model = VertexAIModel('gemini-1.5-flash', region='asia-east1')
agent = Agent(model)
...
```

[`VertexAiRegion`](https://ai.pydantic.dev/api/models/vertexai/#pydantic_ai.models.vertexai.VertexAiRegion) contains a list of available regions.

## Groq

### Install

To use [`GroqModel`](https://ai.pydantic.dev/api/models/groq/#pydantic_ai.models.groq.GroqModel), you need to either install [`pydantic-ai`](https://ai.pydantic.dev/install/), or install [`pydantic-ai-slim`](https://ai.pydantic.dev/install/#slim-install) with the `groq` optional group:

```
pip install 'pydantic-ai-slim[groq]'
```

```
uv add 'pydantic-ai-slim[groq]'
```

### Configuration

To use [Groq](https://groq.com/) through their API, go to [console.groq.com/keys](https://console.groq.com/keys) and follow your nose until you find the place to generate an API key.

[`GroqModelName`](https://ai.pydantic.dev/api/models/groq/#pydantic_ai.models.groq.GroqModelName) contains a list of available Groq models.

### Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export GROQ_API_KEY='your-api-key'
```

You can then use [`GroqModel`](https://ai.pydantic.dev/api/models/groq/#pydantic_ai.models.groq.GroqModel) by name:

groq\_model\_by\_name.py

```python
from pydantic_ai import Agent

agent = Agent('groq:llama-3.3-70b-versatile')
...
```

Or initialise the model directly with just the model name:

groq\_model\_init.py

```python
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel

model = GroqModel('llama-3.3-70b-versatile')
agent = Agent(model)
...
```

### `api_key` argument

If you don't want to or can't set the environment variable, you can pass it at runtime via the [`api_key` argument](https://ai.pydantic.dev/api/models/groq/#pydantic_ai.models.groq.GroqModel.__init__):

groq\_model\_api\_key.py

```python
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel

model = GroqModel('llama-3.3-70b-versatile', api_key='your-api-key')
agent = Agent(model)
...
```

## Mistral

### Install

To use [`MistralModel`](https://ai.pydantic.dev/api/models/mistral/#pydantic_ai.models.mistral.MistralModel), you need to either install [`pydantic-ai`](https://ai.pydantic.dev/install/), or install [`pydantic-ai-slim`](https://ai.pydantic.dev/install/#slim-install) with the `mistral` optional group:

```
pip install 'pydantic-ai-slim[mistral]'
```

```
uv add 'pydantic-ai-slim[mistral]'
```

### Configuration

To use [Mistral](https://mistral.ai/) through their API, go to [console.mistral.ai/api-keys/](https://console.mistral.ai/api-keys/) and follow your nose until you find the place to generate an API key.

[`MistralModelName`](https://ai.pydantic.dev/api/models/mistral/#pydantic_ai.models.mistral.MistralModelName) contains a list of the most popular Mistral models.

### Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export MISTRAL_API_KEY='your-api-key'
```

You can then use [`MistralModel`](https://ai.pydantic.dev/api/models/mistral/#pydantic_ai.models.mistral.MistralModel) by name:

mistral\_model\_by\_name.py

```python
from pydantic_ai import Agent

agent = Agent('mistral:mistral-large-latest')
...
```

Or initialise the model directly with just the model name:

mistral\_model\_init.py

```python
from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel

model = MistralModel('mistral-small-latest')
agent = Agent(model)
...
```

### `api_key` argument

If you don't want to or can't set the environment variable, you can pass it at runtime via the [`api_key` argument](https://ai.pydantic.dev/api/models/mistral/#pydantic_ai.models.mistral.MistralModel.__init__):

mistral\_model\_api\_key.py

```python
from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel

model = MistralModel('mistral-small-latest', api_key='your-api-key')
agent = Agent(model)
...
```

## Cohere

### Install

To use [`CohereModel`](https://ai.pydantic.dev/api/models/cohere/#pydantic_ai.models.cohere.CohereModel), you need to either install [`pydantic-ai`](https://ai.pydantic.dev/install/), or install [`pydantic-ai-slim`](https://ai.pydantic.dev/install/#slim-install) with the `cohere` optional group:

```
pip install 'pydantic-ai-slim[cohere]'
```

```
uv add 'pydantic-ai-slim[cohere]'
```

### Configuration

To use [Cohere](https://cohere.com/) through their API, go to [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys) and follow your nose until you find the place to generate an API key.

[`CohereModelName`](https://ai.pydantic.dev/api/models/cohere/#pydantic_ai.models.cohere.CohereModelName) contains a list of the most popular Cohere models.

### Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export CO_API_KEY='your-api-key'
```

You can then use [`CohereModel`](https://ai.pydantic.dev/api/models/cohere/#pydantic_ai.models.cohere.CohereModel) by name:

cohere\_model\_by\_name.py

```python
from pydantic_ai import Agent

agent = Agent('cohere:command')
...
```

Or initialise the model directly with just the model name:

cohere\_model\_init.py

```python
from pydantic_ai import Agent
from pydantic_ai.models.cohere import CohereModel

model = CohereModel('command', api_key='your-api-key')
agent = Agent(model)
...
```

### `api_key` argument

If you don't want to or can't set the environment variable, you can pass it at runtime via the [`api_key` argument](https://ai.pydantic.dev/api/models/cohere/#pydantic_ai.models.cohere.CohereModel.__init__):

cohere\_model\_api\_key.py

```python
from pydantic_ai import Agent
from pydantic_ai.models.cohere import CohereModel

model = CohereModel('command', api_key='your-api-key')
agent = Agent(model)
...
```

## OpenAI-compatible Models

Many of the models are compatible with OpenAI API, and thus can be used with [`OpenAIModel`](https://ai.pydantic.dev/api/models/openai/#pydantic_ai.models.openai.OpenAIModel) in PydanticAI. Before getting started, check the [OpenAI](https://ai.pydantic.dev/models/#openai) section for installation and configuration instructions.

To use another OpenAI-compatible API, you can make use of the [`base_url`](https://ai.pydantic.dev/api/models/openai/#pydantic_ai.models.openai.OpenAIModel.__init__) and [`api_key`](https://ai.pydantic.dev/api/models/openai/#pydantic_ai.models.openai.OpenAIModel.__init__) arguments:

openai\_model\_base\_url.py

```python
from pydantic_ai.models.openai import OpenAIModel

model = OpenAIModel(
    'model_name',
    base_url='https://<openai-compatible-api-endpoint>.com',
    api_key='your-api-key',
)
...
```

### Ollama

To use [Ollama](https://ollama.com/), you must first download the Ollama client, and then download a model using the [Ollama model library](https://ollama.com/library).

You must also ensure the Ollama server is running when trying to make requests to it. For more information, please see the [Ollama documentation](https://github.com/ollama/ollama/tree/main/docs).

#### Example local usage

With `ollama` installed, you can run the server with the model you want to use:

terminal-run-ollama

```bash
ollama run llama3.2
```

(this will pull the `llama3.2` model if you don't already have it downloaded)

Then run your code, here's a minimal example:

ollama\_example.py

```python
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

class CityLocation(BaseModel):
    city: str
    country: str

ollama_model = OpenAIModel(model_name='llama3.2', base_url='http://localhost:11434/v1')
agent = Agent(ollama_model, result_type=CityLocation)

result = agent.run_sync('Where were the olympics held in 2012?')
print(result.data)
#> city='London' country='United Kingdom'
print(result.usage())
"""
Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65, details=None)
"""
```

#### Example using a remote server

ollama\_example\_with\_remote\_server.py

```python
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

ollama_model = OpenAIModel(
    model_name='qwen2.5-coder:7b',  
    base_url='http://192.168.1.74:11434/v1',  
)

class CityLocation(BaseModel):
    city: str
    country: str

agent = Agent(model=ollama_model, result_type=CityLocation)

result = agent.run_sync('Where were the olympics held in 2012?')
print(result.data)
#> city='London' country='United Kingdom'
print(result.usage())
"""
Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65, details=None)
"""
```

### OpenRouter

To use [OpenRouter](https://openrouter.ai/), first create an API key at [openrouter.ai/keys](https://openrouter.ai/keys).

Once you have the API key, you can pass it to [`OpenAIModel`](https://ai.pydantic.dev/api/models/openai/#pydantic_ai.models.openai.OpenAIModel) as the `api_key` argument:

openrouter\_model\_init.py

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

model = OpenAIModel(
    'anthropic/claude-3.5-sonnet',
    base_url='https://openrouter.ai/api/v1',
    api_key='your-openrouter-api-key',
)
agent = Agent(model)
...
```

### Grok (xAI)

Go to [xAI API Console](https://console.x.ai/) and create an API key. Once you have the API key, follow the [xAI API Documentation](https://docs.x.ai/docs/overview), and set the `base_url` and `api_key` arguments appropriately:

grok\_model\_init.py

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

model = OpenAIModel(
    'grok-2-1212',
    base_url='https://api.x.ai/v1',
    api_key='your-xai-api-key',
)
agent = Agent(model)
...
```

### DeepSeek

Go to [DeepSeek API Platform](https://platform.deepseek.com/api_keys) and create an API key. Once you have the API key, follow the [DeepSeek API Documentation](https://platform.deepseek.com/docs/api/overview), and set the `base_url` and `api_key` arguments appropriately:

deepseek\_model\_init.py

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

model = OpenAIModel(
    'deepseek-chat',
    base_url='https://api.deepseek.com',
    api_key='your-deepseek-api-key',
)
agent = Agent(model)
...
```

## Implementing Custom Models

To implement support for models not already supported, you will need to subclass the [`Model`](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.Model) abstract base class.

For streaming, you'll also need to implement the following abstract base class:

- [`StreamedResponse`](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.StreamedResponse)

The best place to start is to review the source code for existing implementations, e.g. [`OpenAIModel`](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/models/openai.py).

For details on when we'll accept contributions adding new models to PydanticAI, see the [contributing guidelines](https://ai.pydantic.dev/contributing/#new-model-rules).

---
title: "Dependencies - PydanticAI"
source: "https://ai.pydantic.dev/dependencies/"
author:
published:
created: 2025-02-11
description: "Agent Framework / shim to use Pydantic with LLMs"
tags:
  - "clippings"
---
Version Notice

This documentation is ahead of the last release by [3 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.23...main). You may see documentation for features not yet supported in the latest release [v0.0.23 2025-02-07](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.23).

PydanticAI uses a dependency injection system to provide data and services to your agent's [system prompts](https://ai.pydantic.dev/agents/#system-prompts), [tools](https://ai.pydantic.dev/tools/) and [result validators](https://ai.pydantic.dev/results/#result-validators-functions).

Matching PydanticAI's design philosophy, our dependency system tries to use existing best practice in Python development rather than inventing esoteric "magic", this should make dependencies type-safe, understandable easier to test and ultimately easier to deploy in production.

## Defining Dependencies

Dependencies can be any python type. While in simple cases you might be able to pass a single object as a dependency (e.g. an HTTP connection), [dataclasses](https://docs.python.org/3/library/dataclasses.html#module-dataclasses) are generally a convenient container when your dependencies included multiple objects.

Here's an example of defining an agent that requires dependencies.

(**Note:** dependencies aren't actually used in this example, see [Accessing Dependencies](https://ai.pydantic.dev/dependencies/#accessing-dependencies) below)

unused\_dependencies.py

```python
from dataclasses import dataclass

import httpx

from pydantic_ai import Agent

@dataclass
class MyDeps:  
    api_key: str
    http_client: httpx.AsyncClient

agent = Agent(
    'openai:gpt-4o',
    deps_type=MyDeps,  
)

async def main():
    async with httpx.AsyncClient() as client:
        deps = MyDeps('foobar', client)
        result = await agent.run(
            'Tell me a joke.',
            deps=deps,  
        )
        print(result.data)
        #> Did you hear about the toothpaste scandal? They called it Colgate.
```

*(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)*

## Accessing Dependencies

Dependencies are accessed through the [`RunContext`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext) type, this should be the first parameter of system prompt functions etc.

system\_prompt\_dependencies.py

```python
from dataclasses import dataclass

import httpx

from pydantic_ai import Agent, RunContext

@dataclass
class MyDeps:
    api_key: str
    http_client: httpx.AsyncClient

agent = Agent(
    'openai:gpt-4o',
    deps_type=MyDeps,
)

@agent.system_prompt  
async def get_system_prompt(ctx: RunContext[MyDeps]) -> str:  
    response = await ctx.deps.http_client.get(  
        'https://example.com',
        headers={'Authorization': f'Bearer {ctx.deps.api_key}'},  
    )
    response.raise_for_status()
    return f'Prompt: {response.text}'

async def main():
    async with httpx.AsyncClient() as client:
        deps = MyDeps('foobar', client)
        result = await agent.run('Tell me a joke.', deps=deps)
        print(result.data)
        #> Did you hear about the toothpaste scandal? They called it Colgate.
```

*(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)*

### Asynchronous vs. Synchronous dependencies

[System prompt functions](https://ai.pydantic.dev/agents/#system-prompts), [function tools](https://ai.pydantic.dev/tools/) and [result validators](https://ai.pydantic.dev/results/#result-validators-functions) are all run in the async context of an agent run.

If these functions are not coroutines (e.g. `async def`) they are called with [`run_in_executor`](https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor) in a thread pool, it's therefore marginally preferable to use `async` methods where dependencies perform IO, although synchronous dependencies should work fine too.

`run` vs. `run_sync` and Asynchronous vs. Synchronous dependencies

Whether you use synchronous or asynchronous dependencies, is completely independent of whether you use `run` or `run_sync` — `run_sync` is just a wrapper around `run` and agents are always run in an async context.

Here's the same example as above, but with a synchronous dependency:

sync\_dependencies.py

```python
from dataclasses import dataclass

import httpx

from pydantic_ai import Agent, RunContext

@dataclass
class MyDeps:
    api_key: str
    http_client: httpx.Client  

agent = Agent(
    'openai:gpt-4o',
    deps_type=MyDeps,
)

@agent.system_prompt
def get_system_prompt(ctx: RunContext[MyDeps]) -> str:  
    response = ctx.deps.http_client.get(
        'https://example.com', headers={'Authorization': f'Bearer {ctx.deps.api_key}'}
    )
    response.raise_for_status()
    return f'Prompt: {response.text}'

async def main():
    deps = MyDeps('foobar', httpx.Client())
    result = await agent.run(
        'Tell me a joke.',
        deps=deps,
    )
    print(result.data)
    #> Did you hear about the toothpaste scandal? They called it Colgate.
```

*(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)*

## Full Example

As well as system prompts, dependencies can be used in [tools](https://ai.pydantic.dev/tools/) and [result validators](https://ai.pydantic.dev/results/#result-validators-functions).

full\_example.py

```python
from dataclasses import dataclass

import httpx

from pydantic_ai import Agent, ModelRetry, RunContext

@dataclass
class MyDeps:
    api_key: str
    http_client: httpx.AsyncClient

agent = Agent(
    'openai:gpt-4o',
    deps_type=MyDeps,
)

@agent.system_prompt
async def get_system_prompt(ctx: RunContext[MyDeps]) -> str:
    response = await ctx.deps.http_client.get('https://example.com')
    response.raise_for_status()
    return f'Prompt: {response.text}'

@agent.tool  
async def get_joke_material(ctx: RunContext[MyDeps], subject: str) -> str:
    response = await ctx.deps.http_client.get(
        'https://example.com#jokes',
        params={'subject': subject},
        headers={'Authorization': f'Bearer {ctx.deps.api_key}'},
    )
    response.raise_for_status()
    return response.text

@agent.result_validator  
async def validate_result(ctx: RunContext[MyDeps], final_response: str) -> str:
    response = await ctx.deps.http_client.post(
        'https://example.com#validate',
        headers={'Authorization': f'Bearer {ctx.deps.api_key}'},
        params={'query': final_response},
    )
    if response.status_code == 400:
        raise ModelRetry(f'invalid response: {response.text}')
    response.raise_for_status()
    return final_response

async def main():
    async with httpx.AsyncClient() as client:
        deps = MyDeps('foobar', client)
        result = await agent.run('Tell me a joke.', deps=deps)
        print(result.data)
        #> Did you hear about the toothpaste scandal? They called it Colgate.
```

*(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)*

## Overriding Dependencies

When testing agents, it's useful to be able to customise dependencies.

While this can sometimes be done by calling the agent directly within unit tests, we can also override dependencies while calling application code which in turn calls the agent.

This is done via the [`override`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.override) method on the agent.

joke\_app.py

```python
from dataclasses import dataclass

import httpx

from pydantic_ai import Agent, RunContext

@dataclass
class MyDeps:
    api_key: str
    http_client: httpx.AsyncClient

    async def system_prompt_factory(self) -> str:  
        response = await self.http_client.get('https://example.com')
        response.raise_for_status()
        return f'Prompt: {response.text}'

joke_agent = Agent('openai:gpt-4o', deps_type=MyDeps)

@joke_agent.system_prompt
async def get_system_prompt(ctx: RunContext[MyDeps]) -> str:
    return await ctx.deps.system_prompt_factory()  

async def application_code(prompt: str) -> str:  
    ...
    ...
    # now deep within application code we call our agent
    async with httpx.AsyncClient() as client:
        app_deps = MyDeps('foobar', client)
        result = await joke_agent.run(prompt, deps=app_deps)  
    return result.data
```

*(This example is complete, it can be run "as is")*

test\_joke\_app.py

```python
from joke_app import MyDeps, application_code, joke_agent

class TestMyDeps(MyDeps):  
    async def system_prompt_factory(self) -> str:
        return 'test prompt'

async def test_application_code():
    test_deps = TestMyDeps('test_key', None)  
    with joke_agent.override(deps=test_deps):  
        joke = await application_code('Tell me a joke.')  
    assert joke.startswith('Did you hear about the toothpaste scandal?')
```

## Examples

The following examples demonstrate how to use dependencies in PydanticAI:

- [Weather Agent](https://ai.pydantic.dev/examples/weather-agent/)
- [SQL Generation](https://ai.pydantic.dev/examples/sql-gen/)
- [RAG](https://ai.pydantic.dev/examples/rag/)

---
title: "Function Tools - PydanticAI"
source: "https://ai.pydantic.dev/tools/"
author:
published:
created: 2025-02-11
description: "Agent Framework / shim to use Pydantic with LLMs"
tags:
  - "clippings"
---
Version Notice

This documentation is ahead of the last release by [3 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.23...main). You may see documentation for features not yet supported in the latest release [v0.0.23 2025-02-07](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.23).

Function tools provide a mechanism for models to retrieve extra information to help them generate a response.

They're useful when it is impractical or impossible to put all the context an agent might need into the system prompt, or when you want to make agents' behavior more deterministic or reliable by deferring some of the logic required to generate a response to another (not necessarily AI-powered) tool.

Function tools vs. RAG

Function tools are basically the "R" of RAG (Retrieval-Augmented Generation) — they augment what the model can do by letting it request extra information.

The main semantic difference between PydanticAI Tools and RAG is RAG is synonymous with vector search, while PydanticAI tools are more general-purpose. (Note: we may add support for vector search functionality in the future, particularly an API for generating embeddings. See [#58](https://github.com/pydantic/pydantic-ai/issues/58))

There are a number of ways to register tools with an agent:

- via the [`@agent.tool`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.tool) decorator — for tools that need access to the agent [context](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext)
- via the [`@agent.tool_plain`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.tool_plain) decorator — for tools that do not need access to the agent [context](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext)
- via the [`tools`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.__init__) keyword argument to `Agent` which can take either plain functions, or instances of [`Tool`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.Tool)

`@agent.tool` is considered the default decorator since in the majority of cases tools will need access to the agent context.

Here's an example using both:

dice\_game.py

```python
import random

from pydantic_ai import Agent, RunContext

agent = Agent(
    'google-gla:gemini-1.5-flash',  
    deps_type=str,  
    system_prompt=(
        "You're a dice game, you should roll the die and see if the number "
        "you get back matches the user's guess. If so, tell them they're a winner. "
        "Use the player's name in the response."
    ),
)

@agent.tool_plain  
def roll_die() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 6))

@agent.tool  
def get_player_name(ctx: RunContext[str]) -> str:
    """Get the player's name."""
    return ctx.deps

dice_result = agent.run_sync('My guess is 4', deps='Anne')  
print(dice_result.data)
#> Congratulations Anne, you guessed correctly! You're a winner!
```

*(This example is complete, it can be run "as is")*

Let's print the messages from that game to see what happened:

dice\_game\_messages.py

```python
from dice_game import dice_result

print(dice_result.all_messages())
"""
[
    ModelRequest(
        parts=[
            SystemPromptPart(
                content="You're a dice game, you should roll the die and see if the number you get back matches the user's guess. If so, tell them they're a winner. Use the player's name in the response.",
                dynamic_ref=None,
                part_kind='system-prompt',
            ),
            UserPromptPart(
                content='My guess is 4',
                timestamp=datetime.datetime(...),
                part_kind='user-prompt',
            ),
        ],
        kind='request',
    ),
    ModelResponse(
        parts=[
            ToolCallPart(
                tool_name='roll_die', args={}, tool_call_id=None, part_kind='tool-call'
            )
        ],
        model_name='function:model_logic',
        timestamp=datetime.datetime(...),
        kind='response',
    ),
    ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name='roll_die',
                content='4',
                tool_call_id=None,
                timestamp=datetime.datetime(...),
                part_kind='tool-return',
            )
        ],
        kind='request',
    ),
    ModelResponse(
        parts=[
            ToolCallPart(
                tool_name='get_player_name',
                args={},
                tool_call_id=None,
                part_kind='tool-call',
            )
        ],
        model_name='function:model_logic',
        timestamp=datetime.datetime(...),
        kind='response',
    ),
    ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name='get_player_name',
                content='Anne',
                tool_call_id=None,
                timestamp=datetime.datetime(...),
                part_kind='tool-return',
            )
        ],
        kind='request',
    ),
    ModelResponse(
        parts=[
            TextPart(
                content="Congratulations Anne, you guessed correctly! You're a winner!",
                part_kind='text',
            )
        ],
        model_name='function:model_logic',
        timestamp=datetime.datetime(...),
        kind='response',
    ),
]
"""
```

We can represent this with a diagram:

As well as using the decorators, we can register tools via the `tools` argument to the [`Agent` constructor](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.__init__). This is useful when you want to reuse tools, and can also give more fine-grained control over the tools.

dice\_game\_tool\_kwarg.py

```python
import random

from pydantic_ai import Agent, RunContext, Tool

def roll_die() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 6))

def get_player_name(ctx: RunContext[str]) -> str:
    """Get the player's name."""
    return ctx.deps

agent_a = Agent(
    'google-gla:gemini-1.5-flash',
    deps_type=str,
    tools=[roll_die, get_player_name],  
)
agent_b = Agent(
    'google-gla:gemini-1.5-flash',
    deps_type=str,
    tools=[  
        Tool(roll_die, takes_ctx=False),
        Tool(get_player_name, takes_ctx=True),
    ],
)
dice_result = agent_b.run_sync('My guess is 4', deps='Anne')
print(dice_result.data)
#> Congratulations Anne, you guessed correctly! You're a winner!
```

*(This example is complete, it can be run "as is")*

As the name suggests, function tools use the model's "tools" or "functions" API to let the model know what is available to call. Tools or functions are also used to define the schema(s) for structured responses, thus a model might have access to many tools, some of which call function tools while others end the run and return a result.

Function parameters are extracted from the function signature, and all parameters except `RunContext` are used to build the schema for that tool call.

Even better, PydanticAI extracts the docstring from functions and (thanks to [griffe](https://mkdocstrings.github.io/griffe/)) extracts parameter descriptions from the docstring and adds them to the schema.

[Griffe supports](https://mkdocstrings.github.io/griffe/reference/docstrings/#docstrings) extracting parameter descriptions from `google`, `numpy`, and `sphinx` style docstrings. PydanticAI will infer the format to use based on the docstring, but you can explicitly set it using [`docstring_format`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.DocstringFormat). You can also enforce parameter requirements by setting `require_parameter_descriptions=True`. This will raise a [`UserError`](https://ai.pydantic.dev/api/exceptions/#pydantic_ai.exceptions.UserError) if a parameter description is missing.

To demonstrate a tool's schema, here we use [`FunctionModel`](https://ai.pydantic.dev/api/models/function/#pydantic_ai.models.function.FunctionModel) to print the schema a model would receive:

tool\_schema.py

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

agent = Agent()

@agent.tool_plain(docstring_format='google', require_parameter_descriptions=True)
def foobar(a: int, b: str, c: dict[str, list[float]]) -> str:
    """Get me foobar.

    Args:
        a: apple pie
        b: banana cake
        c: carrot smoothie
    """
    return f'{a} {b} {c}'

def print_schema(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    tool = info.function_tools[0]
    print(tool.description)
    #> Get me foobar.
    print(tool.parameters_json_schema)
    """
    {
        'properties': {
            'a': {'description': 'apple pie', 'title': 'A', 'type': 'integer'},
            'b': {'description': 'banana cake', 'title': 'B', 'type': 'string'},
            'c': {
                'additionalProperties': {'items': {'type': 'number'}, 'type': 'array'},
                'description': 'carrot smoothie',
                'title': 'C',
                'type': 'object',
            },
        },
        'required': ['a', 'b', 'c'],
        'type': 'object',
        'additionalProperties': False,
    }
    """
    return ModelResponse(parts=[TextPart('foobar')])

agent.run_sync('hello', model=FunctionModel(print_schema))
```

*(This example is complete, it can be run "as is")*

The return type of tool can be anything which Pydantic can serialize to JSON as some models (e.g. Gemini) support semi-structured return values, some expect text (OpenAI) but seem to be just as good at extracting meaning from the data. If a Python object is returned and the model expects a string, the value will be serialized to JSON.

If a tool has a single parameter that can be represented as an object in JSON schema (e.g. dataclass, TypedDict, pydantic model), the schema for the tool is simplified to be just that object.

Here's an example where we use [`TestModel.last_model_request_parameters`](https://ai.pydantic.dev/api/models/test/#pydantic_ai.models.test.TestModel.last_model_request_parameters) to inspect the tool schema that would be passed to the model.

single\_parameter\_tool.py

```python
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

agent = Agent()

class Foobar(BaseModel):
    """This is a Foobar"""

    x: int
    y: str
    z: float = 3.14

@agent.tool_plain
def foobar(f: Foobar) -> str:
    return str(f)

test_model = TestModel()
result = agent.run_sync('hello', model=test_model)
print(result.data)
#> {"foobar":"x=0 y='a' z=3.14"}
print(test_model.last_model_request_parameters.function_tools)
"""
[
    ToolDefinition(
        name='foobar',
        description='This is a Foobar',
        parameters_json_schema={
            'properties': {
                'x': {'title': 'X', 'type': 'integer'},
                'y': {'title': 'Y', 'type': 'string'},
                'z': {'default': 3.14, 'title': 'Z', 'type': 'number'},
            },
            'required': ['x', 'y'],
            'title': 'Foobar',
            'type': 'object',
        },
        outer_typed_dict_key=None,
    )
]
"""
```

*(This example is complete, it can be run "as is")*

Tools can optionally be defined with another function: `prepare`, which is called at each step of a run to customize the definition of the tool passed to the model, or omit the tool completely from that step.

A `prepare` method can be registered via the `prepare` kwarg to any of the tool registration mechanisms:

- [`@agent.tool`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.tool) decorator
- [`@agent.tool_plain`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.tool_plain) decorator
- [`Tool`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.Tool) dataclass

The `prepare` method, should be of type [`ToolPrepareFunc`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolPrepareFunc), a function which takes [`RunContext`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext) and a pre-built [`ToolDefinition`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolDefinition), and should either return that `ToolDefinition` with or without modifying it, return a new `ToolDefinition`, or return `None` to indicate this tools should not be registered for that step.

Here's a simple `prepare` method that only includes the tool if the value of the dependency is `42`.

As with the previous example, we use [`TestModel`](https://ai.pydantic.dev/api/models/test/#pydantic_ai.models.test.TestModel) to demonstrate the behavior without calling a real model.

tool\_only\_if\_42.py

```python
from typing import Union

from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import ToolDefinition

agent = Agent('test')

async def only_if_42(
    ctx: RunContext[int], tool_def: ToolDefinition
) -> Union[ToolDefinition, None]:
    if ctx.deps == 42:
        return tool_def

@agent.tool(prepare=only_if_42)
def hitchhiker(ctx: RunContext[int], answer: str) -> str:
    return f'{ctx.deps} {answer}'

result = agent.run_sync('testing...', deps=41)
print(result.data)
#> success (no tool calls)
result = agent.run_sync('testing...', deps=42)
print(result.data)
#> {"hitchhiker":"42 a"}
```

*(This example is complete, it can be run "as is")*

Here's a more complex example where we change the description of the `name` parameter to based on the value of `deps`

For the sake of variation, we create this tool using the [`Tool`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.Tool) dataclass.

customize\_name.py

```python
from __future__ import annotations

from typing import Literal

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import Tool, ToolDefinition

def greet(name: str) -> str:
    return f'hello {name}'

async def prepare_greet(
    ctx: RunContext[Literal['human', 'machine']], tool_def: ToolDefinition
) -> ToolDefinition | None:
    d = f'Name of the {ctx.deps} to greet.'
    tool_def.parameters_json_schema['properties']['name']['description'] = d
    return tool_def

greet_tool = Tool(greet, prepare=prepare_greet)
test_model = TestModel()
agent = Agent(test_model, tools=[greet_tool], deps_type=Literal['human', 'machine'])

result = agent.run_sync('testing...', deps='human')
print(result.data)
#> {"greet":"hello a"}
print(test_model.last_model_request_parameters.function_tools)
"""
[
    ToolDefinition(
        name='greet',
        description='',
        parameters_json_schema={
            'properties': {
                'name': {
                    'title': 'Name',
                    'type': 'string',
                    'description': 'Name of the human to greet.',
                }
            },
            'required': ['name'],
            'type': 'object',
            'additionalProperties': False,
        },
        outer_typed_dict_key=None,
    )
]
"""
```

*(This example is complete, it can be run "as is")*

---
title: "Results - PydanticAI"
source: "https://ai.pydantic.dev/results/"
author:
published:
created: 2025-02-11
description: "Agent Framework / shim to use Pydantic with LLMs"
tags:
  - "clippings"
---
Version Notice

This documentation is ahead of the last release by [3 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.23...main). You may see documentation for features not yet supported in the latest release [v0.0.23 2025-02-07](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.23).

Results are the final values returned from [running an agent](https://ai.pydantic.dev/agents/#running-agents). The result values are wrapped in [`RunResult`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.RunResult) and [`StreamedRunResult`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.StreamedRunResult) so you can access other data like [usage](https://ai.pydantic.dev/api/usage/#pydantic_ai.usage.Usage) of the run and [message history](https://ai.pydantic.dev/message-history/#accessing-messages-from-results)

Both `RunResult` and `StreamedRunResult` are generic in the data they wrap, so typing information about the data returned by the agent is preserved.

olympics.py

```python
from pydantic import BaseModel

from pydantic_ai import Agent

class CityLocation(BaseModel):
    city: str
    country: str

agent = Agent('google-gla:gemini-1.5-flash', result_type=CityLocation)
result = agent.run_sync('Where were the olympics held in 2012?')
print(result.data)
#> city='London' country='United Kingdom'
print(result.usage())
"""
Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65, details=None)
"""
```

*(This example is complete, it can be run "as is")*

Runs end when either a plain text response is received or the model calls a tool associated with one of the structured result types. We will add limits to make sure a run doesn't go on indefinitely, see [#70](https://github.com/pydantic/pydantic-ai/issues/70).

## Result data

When the result type is `str`, or a union including `str`, plain text responses are enabled on the model, and the raw text response from the model is used as the response data.

If the result type is a union with multiple members (after remove `str` from the members), each member is registered as a separate tool with the model in order to reduce the complexity of the tool schemas and maximise the chances a model will respond correctly.

If the result type schema is not of type `"object"`, the result type is wrapped in a single element object, so the schema of all tools registered with the model are object schemas.

Structured results (like tools) use Pydantic to build the JSON schema used for the tool, and to validate the data returned by the model.

Bring on PEP-747

Until [PEP-747](https://peps.python.org/pep-0747/) "Annotating Type Forms" lands, unions are not valid as `type`s in Python.

When creating the agent we need to `# type: ignore` the `result_type` argument, and add a type hint to tell type checkers about the type of the agent.

Here's an example of returning either text or a structured value

box\_or\_error.py

```python
from typing import Union

from pydantic import BaseModel

from pydantic_ai import Agent

class Box(BaseModel):
    width: int
    height: int
    depth: int
    units: str

agent: Agent[None, Union[Box, str]] = Agent(
    'openai:gpt-4o-mini',
    result_type=Union[Box, str],  # type: ignore
    system_prompt=(
        "Extract me the dimensions of a box, "
        "if you can't extract all data, ask the user to try again."
    ),
)

result = agent.run_sync('The box is 10x20x30')
print(result.data)
#> Please provide the units for the dimensions (e.g., cm, in, m).

result = agent.run_sync('The box is 10x20x30 cm')
print(result.data)
#> width=10 height=20 depth=30 units='cm'
```

*(This example is complete, it can be run "as is")*

Here's an example of using a union return type which registered multiple tools, and wraps non-object schemas in an object:

colors\_or\_sizes.py

```python
from typing import Union

from pydantic_ai import Agent

agent: Agent[None, Union[list[str], list[int]]] = Agent(
    'openai:gpt-4o-mini',
    result_type=Union[list[str], list[int]],  # type: ignore
    system_prompt='Extract either colors or sizes from the shapes provided.',
)

result = agent.run_sync('red square, blue circle, green triangle')
print(result.data)
#> ['red', 'blue', 'green']

result = agent.run_sync('square size 10, circle size 20, triangle size 30')
print(result.data)
#> [10, 20, 30]
```

*(This example is complete, it can be run "as is")*

### Result validators functions

Some validation is inconvenient or impossible to do in Pydantic validators, in particular when the validation requires IO and is asynchronous. PydanticAI provides a way to add validation functions via the [`agent.result_validator`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.result_validator) decorator.

Here's a simplified variant of the [SQL Generation example](https://ai.pydantic.dev/examples/sql-gen/):

sql\_gen.py

```python
from typing import Union

from fake_database import DatabaseConn, QueryError
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext, ModelRetry

class Success(BaseModel):
    sql_query: str

class InvalidRequest(BaseModel):
    error_message: str

Response = Union[Success, InvalidRequest]
agent: Agent[DatabaseConn, Response] = Agent(
    'google-gla:gemini-1.5-flash',
    result_type=Response,  # type: ignore
    deps_type=DatabaseConn,
    system_prompt='Generate PostgreSQL flavored SQL queries based on user input.',
)

@agent.result_validator
async def validate_result(ctx: RunContext[DatabaseConn], result: Response) -> Response:
    if isinstance(result, InvalidRequest):
        return result
    try:
        await ctx.deps.execute(f'EXPLAIN {result.sql_query}')
    except QueryError as e:
        raise ModelRetry(f'Invalid query: {e}') from e
    else:
        return result

result = agent.run_sync(
    'get me users who were last active yesterday.', deps=DatabaseConn()
)
print(result.data)
#> sql_query='SELECT * FROM users WHERE last_active::date = today() - interval 1 day'
```

*(This example is complete, it can be run "as is")*

## Streamed Results

There two main challenges with streamed results:

1. Validating structured responses before they're complete, this is achieved by "partial validation" which was recently added to Pydantic in [pydantic/pydantic#10748](https://github.com/pydantic/pydantic/pull/10748).
2. When receiving a response, we don't know if it's the final response without starting to stream it and peeking at the content. PydanticAI streams just enough of the response to sniff out if it's a tool call or a result, then streams the whole thing and calls tools, or returns the stream as a [`StreamedRunResult`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.StreamedRunResult).

### Streaming Text

Example of streamed text result:

streamed\_hello\_world.py

```python
from pydantic_ai import Agent

agent = Agent('google-gla:gemini-1.5-flash')  

async def main():
    async with agent.run_stream('Where does "hello world" come from?') as result:  
        async for message in result.stream_text():  
            print(message)
            #> The first known
            #> The first known use of "hello,
            #> The first known use of "hello, world" was in
            #> The first known use of "hello, world" was in a 1974 textbook
            #> The first known use of "hello, world" was in a 1974 textbook about the C
            #> The first known use of "hello, world" was in a 1974 textbook about the C programming language.
```

*(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)*

We can also stream text as deltas rather than the entire text in each item:

streamed\_delta\_hello\_world.py

```python
from pydantic_ai import Agent

agent = Agent('google-gla:gemini-1.5-flash')

async def main():
    async with agent.run_stream('Where does "hello world" come from?') as result:
        async for message in result.stream_text(delta=True):  
            print(message)
            #> The first known
            #> use of "hello,
            #> world" was in
            #> a 1974 textbook
            #> about the C
            #> programming language.
```

*(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)*

Result message not included in `messages`

The final result message will **NOT** be added to result messages if you use `.stream_text(delta=True)`, see [Messages and chat history](https://ai.pydantic.dev/message-history/) for more information.

### Streaming Structured Responses

Not all types are supported with partial validation in Pydantic, see [pydantic/pydantic#10748](https://github.com/pydantic/pydantic/pull/10748), generally for model-like structures it's currently best to use `TypeDict`.

Here's an example of streaming a use profile as it's built:

streamed\_user\_profile.py

```python
from datetime import date

from typing_extensions import TypedDict

from pydantic_ai import Agent

class UserProfile(TypedDict, total=False):
    name: str
    dob: date
    bio: str

agent = Agent(
    'openai:gpt-4o',
    result_type=UserProfile,
    system_prompt='Extract a user profile from the input',
)

async def main():
    user_input = 'My name is Ben, I was born on January 28th 1990, I like the chain the dog and the pyramid.'
    async with agent.run_stream(user_input) as result:
        async for profile in result.stream():
            print(profile)
            #> {'name': 'Ben'}
            #> {'name': 'Ben'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the '}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyr'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyramid'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyramid'}
```

*(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)*

If you want fine-grained control of validation, particularly catching validation errors, you can use the following pattern:

streamed\_user\_profile.py

```python
from datetime import date

from pydantic import ValidationError
from typing_extensions import TypedDict

from pydantic_ai import Agent

class UserProfile(TypedDict, total=False):
    name: str
    dob: date
    bio: str

agent = Agent('openai:gpt-4o', result_type=UserProfile)

async def main():
    user_input = 'My name is Ben, I was born on January 28th 1990, I like the chain the dog and the pyramid.'
    async with agent.run_stream(user_input) as result:
        async for message, last in result.stream_structured(debounce_by=0.01):  
            try:
                profile = await result.validate_structured_result(  
                    message,
                    allow_partial=not last,
                )
            except ValidationError:
                continue
            print(profile)
            #> {'name': 'Ben'}
            #> {'name': 'Ben'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the '}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyr'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyramid'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyramid'}
```

*(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)*

## Examples

The following examples demonstrate how to use streamed responses in PydanticAI:

- [Stream markdown](https://ai.pydantic.dev/examples/stream-markdown/)
- [Stream Whales](https://ai.pydantic.dev/examples/stream-whales/)

---
title: "Messages and chat history - PydanticAI"
source: "https://ai.pydantic.dev/message-history/"
author:
published:
created: 2025-02-11
description: "Agent Framework / shim to use Pydantic with LLMs"
tags:
  - "clippings"
---
Version Notice

This documentation is ahead of the last release by [3 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.23...main). You may see documentation for features not yet supported in the latest release [v0.0.23 2025-02-07](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.23).

PydanticAI provides access to messages exchanged during an agent run. These messages can be used both to continue a coherent conversation, and to understand how an agent performed.

### Accessing Messages from Results

After running an agent, you can access the messages exchanged during that run from the `result` object.

Both [`RunResult`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.RunResult) (returned by [`Agent.run`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run), [`Agent.run_sync`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run_sync)) and [`StreamedRunResult`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.StreamedRunResult) (returned by [`Agent.run_stream`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run_stream)) have the following methods:

- [`all_messages()`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.RunResult.all_messages): returns all messages, including messages from prior runs. There's also a variant that returns JSON bytes, [`all_messages_json()`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.RunResult.all_messages_json).
- [`new_messages()`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.RunResult.new_messages): returns only the messages from the current run. There's also a variant that returns JSON bytes, [`new_messages_json()`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.RunResult.new_messages_json).

Example of accessing methods on a [`RunResult`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.RunResult) :

run\_result\_messages.py

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')

result = agent.run_sync('Tell me a joke.')
print(result.data)
#> Did you hear about the toothpaste scandal? They called it Colgate.

# all messages from the run
print(result.all_messages())
"""
[
    ModelRequest(
        parts=[
            SystemPromptPart(
                content='Be a helpful assistant.',
                dynamic_ref=None,
                part_kind='system-prompt',
            ),
            UserPromptPart(
                content='Tell me a joke.',
                timestamp=datetime.datetime(...),
                part_kind='user-prompt',
            ),
        ],
        kind='request',
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='Did you hear about the toothpaste scandal? They called it Colgate.',
                part_kind='text',
            )
        ],
        model_name='function:model_logic',
        timestamp=datetime.datetime(...),
        kind='response',
    ),
]
"""
```

*(This example is complete, it can be run "as is")*

Example of accessing methods on a [`StreamedRunResult`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.StreamedRunResult) :

streamed\_run\_result\_messages.py

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')

async def main():
    async with agent.run_stream('Tell me a joke.') as result:
        # incomplete messages before the stream finishes
        print(result.all_messages())
        """
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='Be a helpful assistant.',
                        dynamic_ref=None,
                        part_kind='system-prompt',
                    ),
                    UserPromptPart(
                        content='Tell me a joke.',
                        timestamp=datetime.datetime(...),
                        part_kind='user-prompt',
                    ),
                ],
                kind='request',
            )
        ]
        """

        async for text in result.stream_text():
            print(text)
            #> Did you hear
            #> Did you hear about the toothpaste
            #> Did you hear about the toothpaste scandal? They called
            #> Did you hear about the toothpaste scandal? They called it Colgate.

        # complete messages once the stream finishes
        print(result.all_messages())
        """
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='Be a helpful assistant.',
                        dynamic_ref=None,
                        part_kind='system-prompt',
                    ),
                    UserPromptPart(
                        content='Tell me a joke.',
                        timestamp=datetime.datetime(...),
                        part_kind='user-prompt',
                    ),
                ],
                kind='request',
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='Did you hear about the toothpaste scandal? They called it Colgate.',
                        part_kind='text',
                    )
                ],
                model_name='function:stream_model_logic',
                timestamp=datetime.datetime(...),
                kind='response',
            ),
        ]
        """
```

*(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)*

### Using Messages as Input for Further Agent Runs

The primary use of message histories in PydanticAI is to maintain context across multiple agent runs.

To use existing messages in a run, pass them to the `message_history` parameter of [`Agent.run`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run), [`Agent.run_sync`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run_sync) or [`Agent.run_stream`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run_stream).

If `message_history` is set and not empty, a new system prompt is not generated — we assume the existing message history includes a system prompt.

Reusing messages in a conversation

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')

result1 = agent.run_sync('Tell me a joke.')
print(result1.data)
#> Did you hear about the toothpaste scandal? They called it Colgate.

result2 = agent.run_sync('Explain?', message_history=result1.new_messages())
print(result2.data)
#> This is an excellent joke invented by Samuel Colvin, it needs no explanation.

print(result2.all_messages())
"""
[
    ModelRequest(
        parts=[
            SystemPromptPart(
                content='Be a helpful assistant.',
                dynamic_ref=None,
                part_kind='system-prompt',
            ),
            UserPromptPart(
                content='Tell me a joke.',
                timestamp=datetime.datetime(...),
                part_kind='user-prompt',
            ),
        ],
        kind='request',
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='Did you hear about the toothpaste scandal? They called it Colgate.',
                part_kind='text',
            )
        ],
        model_name='function:model_logic',
        timestamp=datetime.datetime(...),
        kind='response',
    ),
    ModelRequest(
        parts=[
            UserPromptPart(
                content='Explain?',
                timestamp=datetime.datetime(...),
                part_kind='user-prompt',
            )
        ],
        kind='request',
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='This is an excellent joke invented by Samuel Colvin, it needs no explanation.',
                part_kind='text',
            )
        ],
        model_name='function:model_logic',
        timestamp=datetime.datetime(...),
        kind='response',
    ),
]
"""
```

*(This example is complete, it can be run "as is")*

## Other ways of using messages

Since messages are defined by simple dataclasses, you can manually create and manipulate, e.g. for testing.

The message format is independent of the model used, so you can use messages in different agents, or the same agent with different models.

In the example below, we reuse the message from the first agent run, which uses the `openai:gpt-4o` model, in a second agent run using the `google-gla:gemini-1.5-pro` model.

Reusing messages with a different model

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')

result1 = agent.run_sync('Tell me a joke.')
print(result1.data)
#> Did you hear about the toothpaste scandal? They called it Colgate.

result2 = agent.run_sync(
    'Explain?',
    model='google-gla:gemini-1.5-pro',
    message_history=result1.new_messages(),
)
print(result2.data)
#> This is an excellent joke invented by Samuel Colvin, it needs no explanation.

print(result2.all_messages())
"""
[
    ModelRequest(
        parts=[
            SystemPromptPart(
                content='Be a helpful assistant.',
                dynamic_ref=None,
                part_kind='system-prompt',
            ),
            UserPromptPart(
                content='Tell me a joke.',
                timestamp=datetime.datetime(...),
                part_kind='user-prompt',
            ),
        ],
        kind='request',
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='Did you hear about the toothpaste scandal? They called it Colgate.',
                part_kind='text',
            )
        ],
        model_name='function:model_logic',
        timestamp=datetime.datetime(...),
        kind='response',
    ),
    ModelRequest(
        parts=[
            UserPromptPart(
                content='Explain?',
                timestamp=datetime.datetime(...),
                part_kind='user-prompt',
            )
        ],
        kind='request',
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='This is an excellent joke invented by Samuel Colvin, it needs no explanation.',
                part_kind='text',
            )
        ],
        model_name='function:model_logic',
        timestamp=datetime.datetime(...),
        kind='response',
    ),
]
"""
```

## Examples

For a more complete example of using messages in conversations, see the [chat app](https://ai.pydantic.dev/examples/chat-app/) example.

---
title: "Testing and Evals - PydanticAI"
source: "https://ai.pydantic.dev/testing-evals/"
author:
published:
created: 2025-02-11
description: "Agent Framework / shim to use Pydantic with LLMs"
tags:
  - "clippings"
---
Version Notice

This documentation is ahead of the last release by [3 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.23...main). You may see documentation for features not yet supported in the latest release [v0.0.23 2025-02-07](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.23).

With PydanticAI and LLM integrations in general, there are two distinct kinds of test:

1. **Unit tests** — tests of your application code, and whether it's behaving correctly
2. **Evals** — tests of the LLM, and how good or bad its responses are

For the most part, these two kinds of tests have pretty separate goals and considerations.

## Unit tests

Unit tests for PydanticAI code are just like unit tests for any other Python code.

Because for the most part they're nothing new, we have pretty well established tools and patterns for writing and running these kinds of tests.

Unless you're really sure you know better, you'll probably want to follow roughly this strategy:

- Use [`pytest`](https://docs.pytest.org/en/stable/) as your test harness
- If you find yourself typing out long assertions, use [inline-snapshot](https://15r10nk.github.io/inline-snapshot/latest/)
- Similarly, [dirty-equals](https://dirty-equals.helpmanual.io/latest/) can be useful for comparing large data structures
- Use [`TestModel`](https://ai.pydantic.dev/api/models/test/#pydantic_ai.models.test.TestModel) or [`FunctionModel`](https://ai.pydantic.dev/api/models/function/#pydantic_ai.models.function.FunctionModel) in place of your actual model to avoid the usage, latency and variability of real LLM calls
- Use [`Agent.override`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.override) to replace your model inside your application logic
- Set [`ALLOW_MODEL_REQUESTS=False`](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.ALLOW_MODEL_REQUESTS) globally to block any requests from being made to non-test models accidentally

### Unit testing with `TestModel`

The simplest and fastest way to exercise most of your application code is using [`TestModel`](https://ai.pydantic.dev/api/models/test/#pydantic_ai.models.test.TestModel), this will (by default) call all tools in the agent, then return either plain text or a structured response depending on the return type of the agent.

`TestModel` is not magic

The "clever" (but not too clever) part of `TestModel` is that it will attempt to generate valid structured data for [function tools](https://ai.pydantic.dev/tools/) and [result types](https://ai.pydantic.dev/results/#structured-result-validation) based on the schema of the registered tools.

There's no ML or AI in `TestModel`, it's just plain old procedural Python code that tries to generate data that satisfies the JSON schema of a tool.

The resulting data won't look pretty or relevant, but it should pass Pydantic's validation in most cases. If you want something more sophisticated, use [`FunctionModel`](https://ai.pydantic.dev/api/models/function/#pydantic_ai.models.function.FunctionModel) and write your own data generation logic.

Let's write unit tests for the following application code:

weather\_app.py

```python
import asyncio
from datetime import date

from pydantic_ai import Agent, RunContext

from fake_database import DatabaseConn  
from weather_service import WeatherService  

weather_agent = Agent(
    'openai:gpt-4o',
    deps_type=WeatherService,
    system_prompt='Providing a weather forecast at the locations the user provides.',
)

@weather_agent.tool
def weather_forecast(
    ctx: RunContext[WeatherService], location: str, forecast_date: date
) -> str:
    if forecast_date < date.today():  
        return ctx.deps.get_historic_weather(location, forecast_date)
    else:
        return ctx.deps.get_forecast(location, forecast_date)

async def run_weather_forecast(  
    user_prompts: list[tuple[str, int]], conn: DatabaseConn
):
    """Run weather forecast for a list of user prompts and save."""
    async with WeatherService() as weather_service:

        async def run_forecast(prompt: str, user_id: int):
            result = await weather_agent.run(prompt, deps=weather_service)
            await conn.store_forecast(user_id, result.data)

        # run all prompts in parallel
        await asyncio.gather(
            *(run_forecast(prompt, user_id) for (prompt, user_id) in user_prompts)
        )
```

Here we have a function that takes a list of `(user_prompt, user_id)` tuples, gets a weather forecast for each prompt, and stores the result in the database.

**We want to test this code without having to mock certain objects or modify our code so we can pass test objects in.**

Here's how we would write tests using [`TestModel`](https://ai.pydantic.dev/api/models/test/#pydantic_ai.models.test.TestModel):

test\_weather\_app.py

```python
from datetime import timezone
import pytest

from dirty_equals import IsNow

from pydantic_ai import models, capture_run_messages
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import (
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    ModelRequest,
)

from fake_database import DatabaseConn
from weather_app import run_weather_forecast, weather_agent

pytestmark = pytest.mark.anyio  
models.ALLOW_MODEL_REQUESTS = False  

async def test_forecast():
    conn = DatabaseConn()
    user_id = 1
    with capture_run_messages() as messages:
        with weather_agent.override(model=TestModel()):  
            prompt = 'What will the weather be like in London on 2024-11-28?'
            await run_weather_forecast([(prompt, user_id)], conn)  

    forecast = await conn.get_forecast(user_id)
    assert forecast == '{"weather_forecast":"Sunny with a chance of rain"}'  

    assert messages == [  
        ModelRequest(
            parts=[
                SystemPromptPart(
                    content='Providing a weather forecast at the locations the user provides.',
                ),
                UserPromptPart(
                    content='What will the weather be like in London on 2024-11-28?',
                    timestamp=IsNow(tz=timezone.utc),  
                ),
            ]
        ),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='weather_forecast',
                    args={
                        'location': 'a',
                        'forecast_date': '2024-01-01',  
                    },
                    tool_call_id=None,
                )
            ],
            model_name='test',
            timestamp=IsNow(tz=timezone.utc),
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='weather_forecast',
                    content='Sunny with a chance of rain',
                    tool_call_id=None,
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ],
        ),
        ModelResponse(
            parts=[
                TextPart(
                    content='{"weather_forecast":"Sunny with a chance of rain"}',
                )
            ],
            model_name='test',
            timestamp=IsNow(tz=timezone.utc),
        ),
    ]
```

### Unit testing with `FunctionModel`

The above tests are a great start, but careful readers will notice that the `WeatherService.get_forecast` is never called since `TestModel` calls `weather_forecast` with a date in the past.

To fully exercise `weather_forecast`, we need to use [`FunctionModel`](https://ai.pydantic.dev/api/models/function/#pydantic_ai.models.function.FunctionModel) to customise how the tools is called.

Here's an example of using `FunctionModel` to test the `weather_forecast` tool with custom inputs

test\_weather\_app2.py

```python
import re

import pytest

from pydantic_ai import models
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from fake_database import DatabaseConn
from weather_app import run_weather_forecast, weather_agent

pytestmark = pytest.mark.anyio
models.ALLOW_MODEL_REQUESTS = False

def call_weather_forecast(  
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    if len(messages) == 1:
        # first call, call the weather forecast tool
        user_prompt = messages[0].parts[-1]
        m = re.search(r'\d{4}-\d{2}-\d{2}', user_prompt.content)
        assert m is not None
        args = {'location': 'London', 'forecast_date': m.group()}  
        return ModelResponse(parts=[ToolCallPart('weather_forecast', args)])
    else:
        # second call, return the forecast
        msg = messages[-1].parts[0]
        assert msg.part_kind == 'tool-return'
        return ModelResponse(parts=[TextPart(f'The forecast is: {msg.content}')])

async def test_forecast_future():
    conn = DatabaseConn()
    user_id = 1
    with weather_agent.override(model=FunctionModel(call_weather_forecast)):  
        prompt = 'What will the weather be like in London on 2032-01-01?'
        await run_weather_forecast([(prompt, user_id)], conn)

    forecast = await conn.get_forecast(user_id)
    assert forecast == 'The forecast is: Rainy with a chance of sun'
```

### Overriding model via pytest fixtures

If you're writing lots of tests that all require model to be overridden, you can use [pytest fixtures](https://docs.pytest.org/en/6.2.x/fixture.html) to override the model with [`TestModel`](https://ai.pydantic.dev/api/models/test/#pydantic_ai.models.test.TestModel) or [`FunctionModel`](https://ai.pydantic.dev/api/models/function/#pydantic_ai.models.function.FunctionModel) in a reusable way.

Here's an example of a fixture that overrides the model with `TestModel`:

tests.py

```python
import pytest
from weather_app import weather_agent

from pydantic_ai.models.test import TestModel

@pytest.fixture
def override_weather_agent():
    with weather_agent.override(model=TestModel()):
        yield

async def test_forecast(override_weather_agent: None):
    ...
    # test code here
```

## Evals

"Evals" refers to evaluating a models performance for a specific application.

Warning

Unlike unit tests, evals are an emerging art/science; anyone who claims to know for sure exactly how your evals should be defined can safely be ignored.

Evals are generally more like benchmarks than unit tests, they never "pass" although they do "fail"; you care mostly about how they change over time.

Since evals need to be run against the real model, then can be slow and expensive to run, you generally won't want to run them in CI for every commit.

### Measuring performance

The hardest part of evals is measuring how well the model has performed.

In some cases (e.g. an agent to generate SQL) there are simple, easy to run tests that can be used to measure performance (e.g. is the SQL valid? Does it return the right results? Does it return just the right results?).

In other cases (e.g. an agent that gives advice on quitting smoking) it can be very hard or impossible to make quantitative measures of performance — in the smoking case you'd really need to run a double-blind trial over months, then wait 40 years and observe health outcomes to know if changes to your prompt were an improvement.

There are a few different strategies you can use to measure performance:

- **End to end, self-contained tests** — like the SQL example, we can test the final result of the agent near-instantly
- **Synthetic self-contained tests** — writing unit test style checks that the output is as expected, checks like `'chewing gum' in response`, while these checks might seem simplistic they can be helpful, one nice characteristic is that it's easy to tell what's wrong when they fail
- **LLMs evaluating LLMs** — using another models, or even the same model with a different prompt to evaluate the performance of the agent (like when the class marks each other's homework because the teacher has a hangover), while the downsides and complexities of this approach are obvious, some think it can be a useful tool in the right circumstances
- **Evals in prod** — measuring the end results of the agent in production, then creating a quantitative measure of performance, so you can easily measure changes over time as you change the prompt or model used, [logfire](https://ai.pydantic.dev/logfire/) can be extremely useful in this case since you can write a custom query to measure the performance of your agent

### System prompt customization

The system prompt is the developer's primary tool in controlling an agent's behavior, so it's often useful to be able to customise the system prompt and see how performance changes. This is particularly relevant when the system prompt contains a list of examples and you want to understand how changing that list affects the model's performance.

Let's assume we have the following app for running SQL generated from a user prompt (this examples omits a lot of details for brevity, see the [SQL gen](https://ai.pydantic.dev/examples/sql-gen/) example for a more complete code):

sql\_app.py

```python
import json
from pathlib import Path
from typing import Union

from pydantic_ai import Agent, RunContext

from fake_database import DatabaseConn

class SqlSystemPrompt:  
    def __init__(
        self, examples: Union[list[dict[str, str]], None] = None, db: str = 'PostgreSQL'
    ):
        if examples is None:
            # if examples aren't provided, load them from file, this is the default
            with Path('examples.json').open('rb') as f:
                self.examples = json.load(f)
        else:
            self.examples = examples

        self.db = db

    def build_prompt(self) -> str:  
        return f"""\
Given the following {self.db} table of records, your job is to
write a SQL query that suits the user's request.

Database schema:
CREATE TABLE records (
  ...
);

{''.join(self.format_example(example) for example in self.examples)}
"""

    @staticmethod
    def format_example(example: dict[str, str]) -> str:  
        return f"""\
<example>
  <request>{example['request']}</request>
  <sql>{example['sql']}</sql>
</example>
"""

sql_agent = Agent(
    'google-gla:gemini-1.5-flash',
    deps_type=SqlSystemPrompt,
)

@sql_agent.system_prompt
async def system_prompt(ctx: RunContext[SqlSystemPrompt]) -> str:
    return ctx.deps.build_prompt()

async def user_search(user_prompt: str) -> list[dict[str, str]]:
    """Search the database based on the user's prompts."""
    ...  
    result = await sql_agent.run(user_prompt, deps=SqlSystemPrompt())
    conn = DatabaseConn()
    return await conn.execute(result.data)
```

`examples.json` looks something like this:

```text
request: show me error records with the tag "foobar"
response: SELECT * FROM records WHERE level = 'error' and 'foobar' = ANY(tags)
```

examples.json

```json
{
  "examples": [
    {
      "request": "Show me all records",
      "sql": "SELECT * FROM records;"
    },
    {
      "request": "Show me all records from 2021",
      "sql": "SELECT * FROM records WHERE date_trunc('year', date) = '2021-01-01';"
    },
    {
      "request": "show me error records with the tag 'foobar'",
      "sql": "SELECT * FROM records WHERE level = 'error' and 'foobar' = ANY(tags);"
    },
    ...
  ]
}
```

Now we want a way to quantify the success of the SQL generation so we can judge how changes to the agent affect its performance.

We can use [`Agent.override`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.override) to replace the system prompt with a custom one that uses a subset of examples, and then run the application code (in this case `user_search`). We also run the actual SQL from the examples and compare the "correct" result from the example SQL to the SQL generated by the agent. (We compare the results of running the SQL rather than the SQL itself since the SQL might be semantically equivalent but written in a different way).

To get a quantitative measure of performance, we assign points to each run as follows:

- **\-100** points if the generated SQL is invalid
- **\-1** point for each row returned by the agent (so returning lots of results is discouraged)
- **+5** points for each row returned by the agent that matches the expected result

We use 5-fold cross-validation to judge the performance of the agent using our existing set of examples.

sql\_app\_evals.py

```python
import json
import statistics
from pathlib import Path
from itertools import chain

from fake_database import DatabaseConn, QueryError
from sql_app import sql_agent, SqlSystemPrompt, user_search

async def main():
    with Path('examples.json').open('rb') as f:
        examples = json.load(f)

    # split examples into 5 folds
    fold_size = len(examples) // 5
    folds = [examples[i : i + fold_size] for i in range(0, len(examples), fold_size)]
    conn = DatabaseConn()
    scores = []

    for i, fold in enumerate(folds):
        fold_score = 0
        # build all other folds into a list of examples
        other_folds = list(chain(*(f for j, f in enumerate(folds) if j != i)))
        # create a new system prompt with the other fold examples
        system_prompt = SqlSystemPrompt(examples=other_folds)

        # override the system prompt with the new one
        with sql_agent.override(deps=system_prompt):
            for case in fold:
                try:
                    agent_results = await user_search(case['request'])
                except QueryError as e:
                    print(f'Fold {i} {case}: {e}')
                    fold_score -= 100
                else:
                    # get the expected results using the SQL from this case
                    expected_results = await conn.execute(case['sql'])

                agent_ids = [r['id'] for r in agent_results]
                # each returned value has a score of -1
                fold_score -= len(agent_ids)
                expected_ids = {r['id'] for r in expected_results}

                # each return value that matches the expected value has a score of 3
                fold_score += 5 * len(set(agent_ids) & expected_ids)

        scores.append(fold_score)

    overall_score = statistics.mean(scores)
    print(f'Overall score: {overall_score:0.2f}')
    #> Overall score: 12.00
```

We can then change the prompt, the model, or the examples and see how the score changes over time.

---
title: "Debugging and Monitoring - PydanticAI"
source: "https://ai.pydantic.dev/logfire/"
author:
published:
created: 2025-02-11
description: "Agent Framework / shim to use Pydantic with LLMs"
tags:
  - "clippings"
---
Version Notice

This documentation is ahead of the last release by [3 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.23...main). You may see documentation for features not yet supported in the latest release [v0.0.23 2025-02-07](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.23).

Applications that use LLMs have some challenges that are well known and understood: LLMs are **slow**, **unreliable** and **expensive**.

These applications also have some challenges that most developers have encountered much less often: LLMs are **fickle** and **non-deterministic**. Subtle changes in a prompt can completely change a model's performance, and there's no `EXPLAIN` query you can run to understand why.

Warning

From a software engineers point of view, you can think of LLMs as the worst database you've ever heard of, but worse.

If LLMs weren't so bloody useful, we'd never touch them.

To build successful applications with LLMs, we need new tools to understand both model performance, and the behavior of applications that rely on them.

LLM Observability tools that just let you understand how your model is performing are useless: making API calls to an LLM is easy, it's building that into an application that's hard.

## Pydantic Logfire

[Pydantic Logfire](https://pydantic.dev/logfire) is an observability platform developed by the team who created and maintain Pydantic and PydanticAI. Logfire aims to let you understand your entire application: Gen AI, classic predictive AI, HTTP traffic, database queries and everything else a modern application needs.

Pydantic Logfire is a commercial product

Logfire is a commercially supported, hosted platform with an extremely generous and perpetual [free tier](https://pydantic.dev/pricing/). You can sign up and start using Logfire in a couple of minutes.

PydanticAI has built-in (but optional) support for Logfire via the [`logfire-api`](https://github.com/pydantic/logfire/tree/main/logfire-api) no-op package.

That means if the `logfire` package is installed and configured, detailed information about agent runs is sent to Logfire. But if the `logfire` package is not installed, there's virtually no overhead and nothing is sent.

Here's an example showing details of running the [Weather Agent](https://ai.pydantic.dev/examples/weather-agent/) in Logfire:

[![Weather Agent Logfire](https://ai.pydantic.dev/img/logfire-weather-agent.png)](https://ai.pydantic.dev/img/logfire-weather-agent.png)

## Using Logfire

To use logfire, you'll need a logfire [account](https://logfire.pydantic.dev/), and logfire installed:

```
pip install 'pydantic-ai[logfire]'
```

```
uv add 'pydantic-ai[logfire]'
```

Then authenticate your local environment with logfire:

And configure a project to send data to:

```
uv run logfire projects new
```

(Or use an existing project with `logfire projects use`)

The last step is to add logfire to your code:

adding\_logfire.py

```python
import logfire

logfire.configure()
```

The [logfire documentation](https://logfire.pydantic.dev/docs/) has more details on how to use logfire, including how to instrument other libraries like [Pydantic](https://logfire.pydantic.dev/docs/integrations/pydantic/), [HTTPX](https://logfire.pydantic.dev/docs/integrations/http-clients/httpx/) and [FastAPI](https://logfire.pydantic.dev/docs/integrations/web-frameworks/fastapi/).

Since Logfire is build on [OpenTelemetry](https://opentelemetry.io/), you can use the Logfire Python SDK to send data to any OpenTelemetry collector.

Once you have logfire set up, there are two primary ways it can help you understand your application:

- **Debugging** — Using the live view to see what's happening in your application in real-time.
- **Monitoring** — Using SQL and dashboards to observe the behavior of your application, Logfire is effectively a SQL database that stores information about how your application is running.

### Debugging

To demonstrate how Logfire can let you visualise the flow of a PydanticAI run, here's the view you get from Logfire while running the [chat app examples](https://ai.pydantic.dev/examples/chat-app/):

### Monitoring Performance

We can also query data with SQL in Logfire to monitor the performance of an application. Here's a real world example of using Logfire to monitor PydanticAI runs inside Logfire itself:

[![Logfire monitoring PydanticAI](https://ai.pydantic.dev/img/logfire-monitoring-pydanticai.png)](https://ai.pydantic.dev/img/logfire-monitoring-pydanticai.png)

### Monitoring HTTPX Requests

In order to monitor HTTPX requests made by models, you can use `logfire`'s [HTTPX](https://logfire.pydantic.dev/docs/integrations/http-clients/httpx/) integration.

Instrumentation is as easy as adding the following three lines to your application:

instrument\_httpx.py

```py
...
import logfire
logfire.configure()  # (1)!
logfire.instrument_httpx()  # (2)!
...
```

\`\`\`

In particular, this can help you to trace specific requests, responses, and headers which might be of particular interest if you're using a custom `httpx` client in your model.

---
title: "Multi-agent Applications - PydanticAI"
source: "https://ai.pydantic.dev/multi-agent-applications/"
author:
published:
created: 2025-02-11
description: "Agent Framework / shim to use Pydantic with LLMs"
tags:
  - "clippings"
---
Version Notice

This documentation is ahead of the last release by [3 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.23...main). You may see documentation for features not yet supported in the latest release [v0.0.23 2025-02-07](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.23).

There are roughly four levels of complexity when building applications with PydanticAI:

1. Single agent workflows — what most of the `pydantic_ai` documentation covers
2. [Agent delegation](https://ai.pydantic.dev/multi-agent-applications/#agent-delegation) — agents using another agent via tools
3. [Programmatic agent hand-off](https://ai.pydantic.dev/multi-agent-applications/#programmatic-agent-hand-off) — one agent runs, then application code calls another agent
4. [Graph based control flow](https://ai.pydantic.dev/graph/) — for the most complex cases, a graph-based state machine can be used to control the execution of multiple agents

Of course, you can combine multiple strategies in a single application.

## Agent delegation

"Agent delegation" refers to the scenario where an agent delegates work to another agent, then takes back control when the delegate agent (the agent called from within a tool) finishes.

Since agents are stateless and designed to be global, you do not need to include the agent itself in agent [dependencies](https://ai.pydantic.dev/dependencies/).

You'll generally want to pass [`ctx.usage`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext.usage) to the [`usage`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run) keyword argument of the delegate agent run so usage within that run counts towards the total usage of the parent agent run.

Multiple models

Agent delegation doesn't need to use the same model for each agent. If you choose to use different models within a run, calculating the monetary cost from the final [`result.usage()`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.RunResult.usage) of the run will not be possible, but you can still use [`UsageLimits`](https://ai.pydantic.dev/api/usage/#pydantic_ai.usage.UsageLimits) to avoid unexpected costs.

agent\_delegation\_simple.py

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits

joke_selection_agent = Agent(  
    'openai:gpt-4o',
    system_prompt=(
        'Use the \`joke_factory\` to generate some jokes, then choose the best. '
        'You must return just a single joke.'
    ),
)
joke_generation_agent = Agent(  
    'google-gla:gemini-1.5-flash', result_type=list[str]
)

@joke_selection_agent.tool
async def joke_factory(ctx: RunContext[None], count: int) -> list[str]:
    r = await joke_generation_agent.run(  
        f'Please generate {count} jokes.',
        usage=ctx.usage,  
    )
    return r.data  

result = joke_selection_agent.run_sync(
    'Tell me a joke.',
    usage_limits=UsageLimits(request_limit=5, total_tokens_limit=300),
)
print(result.data)
#> Did you hear about the toothpaste scandal? They called it Colgate.
print(result.usage())
"""
Usage(
    requests=3, request_tokens=204, response_tokens=24, total_tokens=228, details=None
)
"""
```

*(This example is complete, it can be run "as is")*

The control flow for this example is pretty simple and can be summarised as follows:

### Agent delegation and dependencies

Generally the delegate agent needs to either have the same [dependencies](https://ai.pydantic.dev/dependencies/) as the calling agent, or dependencies which are a subset of the calling agent's dependencies.

Initializing dependencies

We say "generally" above since there's nothing to stop you initializing dependencies within a tool call and therefore using interdependencies in a delegate agent that are not available on the parent, this should often be avoided since it can be significantly slower than reusing connections etc. from the parent agent.

agent\_delegation\_deps.py

```python
from dataclasses import dataclass

import httpx

from pydantic_ai import Agent, RunContext

@dataclass
class ClientAndKey:  
    http_client: httpx.AsyncClient
    api_key: str

joke_selection_agent = Agent(
    'openai:gpt-4o',
    deps_type=ClientAndKey,  
    system_prompt=(
        'Use the \`joke_factory\` tool to generate some jokes on the given subject, '
        'then choose the best. You must return just a single joke.'
    ),
)
joke_generation_agent = Agent(
    'gemini-1.5-flash',
    deps_type=ClientAndKey,  
    result_type=list[str],
    system_prompt=(
        'Use the "get_jokes" tool to get some jokes on the given subject, '
        'then extract each joke into a list.'
    ),
)

@joke_selection_agent.tool
async def joke_factory(ctx: RunContext[ClientAndKey], count: int) -> list[str]:
    r = await joke_generation_agent.run(
        f'Please generate {count} jokes.',
        deps=ctx.deps,  
        usage=ctx.usage,
    )
    return r.data

@joke_generation_agent.tool  
async def get_jokes(ctx: RunContext[ClientAndKey], count: int) -> str:
    response = await ctx.deps.http_client.get(
        'https://example.com',
        params={'count': count},
        headers={'Authorization': f'Bearer {ctx.deps.api_key}'},
    )
    response.raise_for_status()
    return response.text

async def main():
    async with httpx.AsyncClient() as client:
        deps = ClientAndKey(client, 'foobar')
        result = await joke_selection_agent.run('Tell me a joke.', deps=deps)
        print(result.data)
        #> Did you hear about the toothpaste scandal? They called it Colgate.
        print(result.usage())  
        """
        Usage(
            requests=4,
            request_tokens=309,
            response_tokens=32,
            total_tokens=341,
            details=None,
        )
        """
```

*(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)*

This example shows how even a fairly simple agent delegation can lead to a complex control flow:

## Programmatic agent hand-off

"Programmatic agent hand-off" refers to the scenario where multiple agents are called in succession, with application code and/or a human in the loop responsible for deciding which agent to call next.

Here agents don't need to use the same deps.

Here we show two agents used in succession, the first to find a flight and the second to extract the user's seat preference.

programmatic\_handoff.py

```python
from typing import Literal, Union

from pydantic import BaseModel, Field
from rich.prompt import Prompt

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import Usage, UsageLimits

class FlightDetails(BaseModel):
    flight_number: str

class Failed(BaseModel):
    """Unable to find a satisfactory choice."""

flight_search_agent = Agent[None, Union[FlightDetails, Failed]](  
    'openai:gpt-4o',
    result_type=Union[FlightDetails, Failed],  # type: ignore
    system_prompt=(
        'Use the "flight_search" tool to find a flight '
        'from the given origin to the given destination.'
    ),
)

@flight_search_agent.tool  
async def flight_search(
    ctx: RunContext[None], origin: str, destination: str
) -> Union[FlightDetails, None]:
    # in reality, this would call a flight search API or
    # use a browser to scrape a flight search website
    return FlightDetails(flight_number='AK456')

usage_limits = UsageLimits(request_limit=15)  

async def find_flight(usage: Usage) -> Union[FlightDetails, None]:  
    message_history: Union[list[ModelMessage], None] = None
    for _ in range(3):
        prompt = Prompt.ask(
            'Where would you like to fly from and to?',
        )
        result = await flight_search_agent.run(
            prompt,
            message_history=message_history,
            usage=usage,
            usage_limits=usage_limits,
        )
        if isinstance(result.data, FlightDetails):
            return result.data
        else:
            message_history = result.all_messages(
                result_tool_return_content='Please try again.'
            )

class SeatPreference(BaseModel):
    row: int = Field(ge=1, le=30)
    seat: Literal['A', 'B', 'C', 'D', 'E', 'F']

# This agent is responsible for extracting the user's seat selection
seat_preference_agent = Agent[None, Union[SeatPreference, Failed]](  
    'openai:gpt-4o',
    result_type=Union[SeatPreference, Failed],  # type: ignore
    system_prompt=(
        "Extract the user's seat preference. "
        'Seats A and F are window seats. '
        'Row 1 is the front row and has extra leg room. '
        'Rows 14, and 20 also have extra leg room. '
    ),
)

async def find_seat(usage: Usage) -> SeatPreference:  
    message_history: Union[list[ModelMessage], None] = None
    while True:
        answer = Prompt.ask('What seat would you like?')

        result = await seat_preference_agent.run(
            answer,
            message_history=message_history,
            usage=usage,
            usage_limits=usage_limits,
        )
        if isinstance(result.data, SeatPreference):
            return result.data
        else:
            print('Could not understand seat preference. Please try again.')
            message_history = result.all_messages()

async def main():  
    usage: Usage = Usage()

    opt_flight_details = await find_flight(usage)
    if opt_flight_details is not None:
        print(f'Flight found: {opt_flight_details.flight_number}')
        #> Flight found: AK456
        seat_preference = await find_seat(usage)
        print(f'Seat preference: {seat_preference}')
        #> Seat preference: row=1 seat='A'
```

*(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)*

The control flow for this example can be summarised as follows:

## Pydantic Graphs

See the [graph](https://ai.pydantic.dev/graph/) documentation on when and how to use graphs.

## Examples

The following examples demonstrate how to use dependencies in PydanticAI:

- [Flight booking](https://ai.pydantic.dev/examples/flight-booking/)

---
title: "Graphs - PydanticAI"
source: "https://ai.pydantic.dev/graph/"
author:
published:
created: 2025-02-11
description: "Agent Framework / shim to use Pydantic with LLMs"
tags:
  - "clippings"
---
Version Notice

This documentation is ahead of the last release by [3 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.23...main). You may see documentation for features not yet supported in the latest release [v0.0.23 2025-02-07](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.23).

Don't use a nail gun unless you need a nail gun

If PydanticAI [agents](https://ai.pydantic.dev/agents/) are a hammer, and [multi-agent workflows](https://ai.pydantic.dev/multi-agent-applications/) are a sledgehammer, then graphs are a nail gun:

- sure, nail guns look cooler than hammers
- but nail guns take a lot more setup than hammers
- and nail guns don't make you a better builder, they make you a builder with a nail gun
- Lastly, (and at the risk of torturing this metaphor), if you're a fan of medieval tools like mallets and untyped Python, you probably won't like nail guns or our approach to graphs. (But then again, if you're not a fan of type hints in Python, you've probably already bounced off PydanticAI to use one of the toy agent frameworks — good luck, and feel free to borrow my sledgehammer when you realize you need it)

In short, graphs are a powerful tool, but they're not the right tool for every job. Please consider other [multi-agent approaches](https://ai.pydantic.dev/multi-agent-applications/) before proceeding.

If you're not confident a graph-based approach is a good idea, it might be unnecessary.

Graphs and finite state machines (FSMs) are a powerful abstraction to model, execute, control and visualize complex workflows.

Alongside PydanticAI, we've developed `pydantic-graph` — an async graph and state machine library for Python where nodes and edges are defined using type hints.

While this library is developed as part of PydanticAI; it has no dependency on `pydantic-ai` and can be considered as a pure graph-based state machine library. You may find it useful whether or not you're using PydanticAI or even building with GenAI.

`pydantic-graph` is designed for advanced users and makes heavy use of Python generics and types hints. It is not designed to be as beginner-friendly as PydanticAI.

Very Early beta

Graph support was [introduced](https://github.com/pydantic/pydantic-ai/pull/528) in v0.0.19 and is in very earlier beta. The API is subject to change. The documentation is incomplete. The implementation is incomplete.

## Installation

`pydantic-graph` is a required dependency of `pydantic-ai`, and an optional dependency of `pydantic-ai-slim`, see [installation instructions](https://ai.pydantic.dev/install/#slim-install) for more information. You can also install it directly:

```
pip install pydantic-graph
```

## Graph Types

`pydantic-graph` made up of a few key components:

### GraphRunContext

[`GraphRunContext`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.GraphRunContext) — The context for the graph run, similar to PydanticAI's [`RunContext`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext). This holds the state of the graph and dependencies and is passed to nodes when they're run.

`GraphRunContext` is generic in the state type of the graph it's used in, [`StateT`](https://ai.pydantic.dev/api/pydantic_graph/state/#pydantic_graph.state.StateT).

### End

[`End`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.End) — return value to indicate the graph run should end.

`End` is generic in the graph return type of the graph it's used in, [`RunEndT`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT).

### Nodes

Subclasses of [`BaseNode`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode) define nodes for execution in the graph.

Nodes, which are generally [`dataclass`es](https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass), generally consist of:

- fields containing any parameters required/optional when calling the node
- the business logic to execute the node, in the [`run`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode.run) method
- return annotations of the [`run`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode.run) method, which are read by `pydantic-graph` to determine the outgoing edges of the node

Nodes are generic in:

- **state**, which must have the same type as the state of graphs they're included in, [`StateT`](https://ai.pydantic.dev/api/pydantic_graph/state/#pydantic_graph.state.StateT) has a default of `None`, so if you're not using state you can omit this generic parameter, see [stateful graphs](https://ai.pydantic.dev/graph/#stateful-graphs) for more information
- **deps**, which must have the same type as the deps of the graph they're included in, [`DepsT`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.DepsT) has a default of `None`, so if you're not using deps you can omit this generic parameter, see [dependency injection](https://ai.pydantic.dev/graph/#dependency-injection) for more information
- **graph return type** — this only applies if the node returns [`End`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.End). [`RunEndT`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT) has a default of [Never](https://docs.python.org/3/library/typing.html#typing.Never) so this generic parameter can be omitted if the node doesn't return `End`, but must be included if it does.

Here's an example of a start or intermediate node in a graph — it can't end the run as it doesn't return [`End`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.End):

intermediate\_node.py

```py
from dataclasses import dataclass

from pydantic_graph import BaseNode, GraphRunContext

@dataclass
class MyNode(BaseNode[MyState]):  
    foo: int  

    async def run(
        self,
        ctx: GraphRunContext[MyState],  
    ) -> AnotherNode:  
        ...
        return AnotherNode()
```

We could extend `MyNode` to optionally end the run if `foo` is divisible by 5:

intermediate\_or\_end\_node.py

```py
from dataclasses import dataclass

from pydantic_graph import BaseNode, End, GraphRunContext

@dataclass
class MyNode(BaseNode[MyState, None, int]):  
    foo: int

    async def run(
        self,
        ctx: GraphRunContext[MyState],
    ) -> AnotherNode | End[int]:  
        if self.foo % 5 == 0:
            return End(self.foo)
        else:
            return AnotherNode()
```

### Graph

[`Graph`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph) — this is the execution graph itself, made up of a set of [node classes](https://ai.pydantic.dev/graph/#nodes) (i.e., `BaseNode` subclasses).

`Graph` is generic in:

- **state** the state type of the graph, [`StateT`](https://ai.pydantic.dev/api/pydantic_graph/state/#pydantic_graph.state.StateT)
- **deps** the deps type of the graph, [`DepsT`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.DepsT)
- **graph return type** the return type of the graph run, [`RunEndT`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT)

Here's an example of a simple graph:

graph\_example.py

```py
from __future__ import annotations

from dataclasses import dataclass

from pydantic_graph import BaseNode, End, Graph, GraphRunContext

@dataclass
class DivisibleBy5(BaseNode[None, None, int]):  
    foo: int

    async def run(
        self,
        ctx: GraphRunContext,
    ) -> Increment | End[int]:
        if self.foo % 5 == 0:
            return End(self.foo)
        else:
            return Increment(self.foo)

@dataclass
class Increment(BaseNode):  
    foo: int

    async def run(self, ctx: GraphRunContext) -> DivisibleBy5:
        return DivisibleBy5(self.foo + 1)

fives_graph = Graph(nodes=[DivisibleBy5, Increment])  
result, history = fives_graph.run_sync(DivisibleBy5(4))  
print(result)
#> 5
# the full history is quite verbose (see below), so we'll just print the summary
print([item.data_snapshot() for item in history])
#> [DivisibleBy5(foo=4), Increment(foo=4), DivisibleBy5(foo=5), End(data=5)]
```

*(This example is complete, it can be run "as is" with Python 3.10+)*

A [mermaid diagram](https://ai.pydantic.dev/graph/#mermaid-diagrams) for this graph can be generated with the following code:

graph\_example\_diagram.py

```py
from graph_example import DivisibleBy5, fives_graph

fives_graph.mermaid_code(start_node=DivisibleBy5)
```

In order to visualize a graph within a `jupyter-notebook`, `IPython.display` needs to be used:

jupyter\_display\_mermaid.py

```python
from graph_example import DivisibleBy5, fives_graph
from IPython.display import Image, display

display(Image(fives_graph.mermaid_image(start_node=DivisibleBy5)))
```

## Stateful Graphs

The "state" concept in `pydantic-graph` provides an optional way to access and mutate an object (often a `dataclass` or Pydantic model) as nodes run in a graph. If you think of Graphs as a production line, then your state is the engine being passed along the line and built up by each node as the graph is run.

In the future, we intend to extend `pydantic-graph` to provide state persistence with the state recorded after each node is run, see [#695](https://github.com/pydantic/pydantic-ai/issues/695).

Here's an example of a graph which represents a vending machine where the user may insert coins and select a product to purchase.

vending\_machine.py

```python
from __future__ import annotations

from dataclasses import dataclass

from rich.prompt import Prompt

from pydantic_graph import BaseNode, End, Graph, GraphRunContext

@dataclass
class MachineState:  
    user_balance: float = 0.0
    product: str | None = None

@dataclass
class InsertCoin(BaseNode[MachineState]):  
    async def run(self, ctx: GraphRunContext[MachineState]) -> CoinsInserted:  
        return CoinsInserted(float(Prompt.ask('Insert coins')))  

@dataclass
class CoinsInserted(BaseNode[MachineState]):
    amount: float  

    async def run(
        self, ctx: GraphRunContext[MachineState]
    ) -> SelectProduct | Purchase:  
        ctx.state.user_balance += self.amount  
        if ctx.state.product is not None:  
            return Purchase(ctx.state.product)
        else:
            return SelectProduct()

@dataclass
class SelectProduct(BaseNode[MachineState]):
    async def run(self, ctx: GraphRunContext[MachineState]) -> Purchase:
        return Purchase(Prompt.ask('Select product'))

PRODUCT_PRICES = {  
    'water': 1.25,
    'soda': 1.50,
    'crisps': 1.75,
    'chocolate': 2.00,
}

@dataclass
class Purchase(BaseNode[MachineState, None, None]):  
    product: str

    async def run(
        self, ctx: GraphRunContext[MachineState]
    ) -> End | InsertCoin | SelectProduct:
        if price := PRODUCT_PRICES.get(self.product):  
            ctx.state.product = self.product  
            if ctx.state.user_balance >= price:  
                ctx.state.user_balance -= price
                return End(None)
            else:
                diff = price - ctx.state.user_balance
                print(f'Not enough money for {self.product}, need {diff:0.2f} more')
                #> Not enough money for crisps, need 0.75 more
                return InsertCoin()  
        else:
            print(f'No such product: {self.product}, try again')
            return SelectProduct()  

vending_machine_graph = Graph(  
    nodes=[InsertCoin, CoinsInserted, SelectProduct, Purchase]
)

async def main():
    state = MachineState()  
    await vending_machine_graph.run(InsertCoin(), state=state)  
    print(f'purchase successful item={state.product} change={state.user_balance:0.2f}')
    #> purchase successful item=crisps change=0.25
```

*(This example is complete, it can be run "as is" with Python 3.10+ — you'll need to add `asyncio.run(main())` to run `main`)*

A [mermaid diagram](https://ai.pydantic.dev/graph/#mermaid-diagrams) for this graph can be generated with the following code:

vending\_machine\_diagram.py

```py
from vending_machine import InsertCoin, vending_machine_graph

vending_machine_graph.mermaid_code(start_node=InsertCoin)
```

The diagram generated by the above code is:

See [below](https://ai.pydantic.dev/graph/#mermaid-diagrams) for more information on generating diagrams.

## GenAI Example

So far we haven't shown an example of a Graph that actually uses PydanticAI or GenAI at all.

In this example, one agent generates a welcome email to a user and the other agent provides feedback on the email.

This graph has a very simple structure:

genai\_email\_feedback.py

```python
from __future__ import annotations as _annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, EmailStr

from pydantic_ai import Agent
from pydantic_ai.format_as_xml import format_as_xml
from pydantic_ai.messages import ModelMessage
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

@dataclass
class User:
    name: str
    email: EmailStr
    interests: list[str]

@dataclass
class Email:
    subject: str
    body: str

@dataclass
class State:
    user: User
    write_agent_messages: list[ModelMessage] = field(default_factory=list)

email_writer_agent = Agent(
    'google-vertex:gemini-1.5-pro',
    result_type=Email,
    system_prompt='Write a welcome email to our tech blog.',
)

@dataclass
class WriteEmail(BaseNode[State]):
    email_feedback: str | None = None

    async def run(self, ctx: GraphRunContext[State]) -> Feedback:
        if self.email_feedback:
            prompt = (
                f'Rewrite the email for the user:\n'
                f'{format_as_xml(ctx.state.user)}\n'
                f'Feedback: {self.email_feedback}'
            )
        else:
            prompt = (
                f'Write a welcome email for the user:\n'
                f'{format_as_xml(ctx.state.user)}'
            )

        result = await email_writer_agent.run(
            prompt,
            message_history=ctx.state.write_agent_messages,
        )
        ctx.state.write_agent_messages += result.all_messages()
        return Feedback(result.data)

class EmailRequiresWrite(BaseModel):
    feedback: str

class EmailOk(BaseModel):
    pass

feedback_agent = Agent[None, EmailRequiresWrite | EmailOk](
    'openai:gpt-4o',
    result_type=EmailRequiresWrite | EmailOk,  # type: ignore
    system_prompt=(
        'Review the email and provide feedback, email must reference the users specific interests.'
    ),
)

@dataclass
class Feedback(BaseNode[State, None, Email]):
    email: Email

    async def run(
        self,
        ctx: GraphRunContext[State],
    ) -> WriteEmail | End[Email]:
        prompt = format_as_xml({'user': ctx.state.user, 'email': self.email})
        result = await feedback_agent.run(prompt)
        if isinstance(result.data, EmailRequiresWrite):
            return WriteEmail(email_feedback=result.data.feedback)
        else:
            return End(self.email)

async def main():
    user = User(
        name='John Doe',
        email='john.joe@example.com',
        interests=['Haskel', 'Lisp', 'Fortran'],
    )
    state = State(user)
    feedback_graph = Graph(nodes=(WriteEmail, Feedback))
    email, _ = await feedback_graph.run(WriteEmail(), state=state)
    print(email)
    """
    Email(
        subject='Welcome to our tech blog!',
        body='Hello John, Welcome to our tech blog! ...',
    )
    """
```

*(This example is complete, it can be run "as is" with Python 3.10+ — you'll need to add `asyncio.run(main())` to run `main`)*

## Custom Control Flow

In many real-world applications, Graphs cannot run uninterrupted from start to finish — they might require external input, or run over an extended period of time such that a single process cannot execute the entire graph run from start to finish without interruption.

In these scenarios the [`next`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph.next) method can be used to run the graph one node at a time.

In this example, an AI asks the user a question, the user provides an answer, the AI evaluates the answer and ends if the user got it right or asks another question if they got it wrong.

`ai_q_and_a_graph.py` — `question_graph` definition

ai\_q\_and\_a\_graph.py

```python
from __future__ import annotations as _annotations

from dataclasses import dataclass, field

from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from pydantic_ai import Agent
from pydantic_ai.format_as_xml import format_as_xml
from pydantic_ai.messages import ModelMessage

ask_agent = Agent('openai:gpt-4o', result_type=str)

@dataclass
class QuestionState:
    question: str | None = None
    ask_agent_messages: list[ModelMessage] = field(default_factory=list)
    evaluate_agent_messages: list[ModelMessage] = field(default_factory=list)

@dataclass
class Ask(BaseNode[QuestionState]):
    async def run(self, ctx: GraphRunContext[QuestionState]) -> Answer:
        result = await ask_agent.run(
            'Ask a simple question with a single correct answer.',
            message_history=ctx.state.ask_agent_messages,
        )
        ctx.state.ask_agent_messages += result.all_messages()
        ctx.state.question = result.data
        return Answer(result.data)

@dataclass
class Answer(BaseNode[QuestionState]):
    question: str
    answer: str | None = None

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Evaluate:
        assert self.answer is not None
        return Evaluate(self.answer)

@dataclass
class EvaluationResult:
    correct: bool
    comment: str

evaluate_agent = Agent(
    'openai:gpt-4o',
    result_type=EvaluationResult,
    system_prompt='Given a question and answer, evaluate if the answer is correct.',
)

@dataclass
class Evaluate(BaseNode[QuestionState]):
    answer: str

    async def run(
        self,
        ctx: GraphRunContext[QuestionState],
    ) -> End[str] | Reprimand:
        assert ctx.state.question is not None
        result = await evaluate_agent.run(
            format_as_xml({'question': ctx.state.question, 'answer': self.answer}),
            message_history=ctx.state.evaluate_agent_messages,
        )
        ctx.state.evaluate_agent_messages += result.all_messages()
        if result.data.correct:
            return End(result.data.comment)
        else:
            return Reprimand(result.data.comment)

@dataclass
class Reprimand(BaseNode[QuestionState]):
    comment: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Ask:
        print(f'Comment: {self.comment}')
        ctx.state.question = None
        return Ask()

question_graph = Graph(nodes=(Ask, Answer, Evaluate, Reprimand))
```

*(This example is complete, it can be run "as is" with Python 3.10+)*

ai\_q\_and\_a\_run.py

```python
from rich.prompt import Prompt

from pydantic_graph import End, HistoryStep

from ai_q_and_a_graph import Ask, question_graph, QuestionState, Answer

async def main():
    state = QuestionState()  
    node = Ask()  
    history: list[HistoryStep[QuestionState]] = []  
    while True:
        node = await question_graph.next(node, history, state=state)  
        if isinstance(node, Answer):
            node.answer = Prompt.ask(node.question)  
        elif isinstance(node, End):  
            print(f'Correct answer! {node.data}')
            #> Correct answer! Well done, 1 + 1 = 2
            print([e.data_snapshot() for e in history])
            """
            [
                Ask(),
                Answer(question='What is the capital of France?', answer='Vichy'),
                Evaluate(answer='Vichy'),
                Reprimand(comment='Vichy is no longer the capital of France.'),
                Ask(),
                Answer(question='what is 1 + 1?', answer='2'),
                Evaluate(answer='2'),
            ]
            """
            return
        # otherwise just continue
```

*(This example is complete, it can be run "as is" with Python 3.10+ — you'll need to add `asyncio.run(main())` to run `main`)*

A [mermaid diagram](https://ai.pydantic.dev/graph/#mermaid-diagrams) for this graph can be generated with the following code:

ai\_q\_and\_a\_diagram.py

```py
from ai_q_and_a_graph import Ask, question_graph

question_graph.mermaid_code(start_node=Ask)
```

You maybe have noticed that although this examples transfers control flow out of the graph run, we're still using [rich's `Prompt.ask`](https://rich.readthedocs.io/en/stable/reference/prompt.html#rich.prompt.PromptBase.ask) to get user input, with the process hanging while we wait for the user to enter a response. For an example of genuine out-of-process control flow, see the [question graph example](https://ai.pydantic.dev/examples/question-graph/).

## Dependency Injection

As with PydanticAI, `pydantic-graph` supports dependency injection via a generic parameter on [`Graph`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph) and [`BaseNode`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode), and the [`GraphRunContext.deps`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.GraphRunContext.deps) fields.

As an example of dependency injection, let's modify the `DivisibleBy5` example [above](https://ai.pydantic.dev/graph/#graph) to use a [`ProcessPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor) to run the compute load in a separate process (this is a contrived example, `ProcessPoolExecutor` wouldn't actually improve performance in this example):

deps\_example.py

```py
from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from pydantic_graph import BaseNode, End, Graph, GraphRunContext

@dataclass
class GraphDeps:
    executor: ProcessPoolExecutor

@dataclass
class DivisibleBy5(BaseNode[None, None, int]):
    foo: int

    async def run(
        self,
        ctx: GraphRunContext,
    ) -> Increment | End[int]:
        if self.foo % 5 == 0:
            return End(self.foo)
        else:
            return Increment(self.foo)

@dataclass
class Increment(BaseNode):
    foo: int

    async def run(self, ctx: GraphRunContext) -> DivisibleBy5:
        loop = asyncio.get_running_loop()
        compute_result = await loop.run_in_executor(
            ctx.deps.executor,
            self.compute,
        )
        return DivisibleBy5(compute_result)

    def compute(self) -> int:
        return self.foo + 1

fives_graph = Graph(nodes=[DivisibleBy5, Increment])

async def main():
    with ProcessPoolExecutor() as executor:
        deps = GraphDeps(executor)
        result, history = await fives_graph.run(DivisibleBy5(3), deps=deps)
    print(result)
    #> 5
    # the full history is quite verbose (see below), so we'll just print the summary
    print([item.data_snapshot() for item in history])
    """
    [
        DivisibleBy5(foo=3),
        Increment(foo=3),
        DivisibleBy5(foo=4),
        Increment(foo=4),
        DivisibleBy5(foo=5),
        End(data=5),
    ]
    """
```

*(This example is complete, it can be run "as is" with Python 3.10+ — you'll need to add `asyncio.run(main())` to run `main`)*

## Mermaid Diagrams

Pydantic Graph can generate [mermaid](https://mermaid.js.org/) [`stateDiagram-v2`](https://mermaid.js.org/syntax/stateDiagram.html) diagrams for graphs, as shown above.

These diagrams can be generated with:

- [`Graph.mermaid_code`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph.mermaid_code) to generate the mermaid code for a graph
- [`Graph.mermaid_image`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph.mermaid_image) to generate an image of the graph using [mermaid.ink](https://mermaid.ink/)
- [`Graph.mermaid_save`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph.mermaid_save) to generate an image of the graph using [mermaid.ink](https://mermaid.ink/) and save it to a file

Beyond the diagrams shown above, you can also customize mermaid diagrams with the following options:

- [`Edge`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.Edge) allows you to apply a label to an edge
- [`BaseNode.docstring_notes`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode.docstring_notes) and [`BaseNode.get_note`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode.get_note) allows you to add notes to nodes
- The [`highlighted_nodes`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph.mermaid_code) parameter allows you to highlight specific node(s) in the diagram

Putting that together, we can edit the last [`ai_q_and_a_graph.py`](https://ai.pydantic.dev/graph/#custom-control-flow) example to:

- add labels to some edges
- add a note to the `Ask` node
- highlight the `Answer` node
- save the diagram as a `PNG` image to file

ai\_q\_and\_a\_graph\_extra.py

```python
...
from typing import Annotated

from pydantic_graph import BaseNode, End, Graph, GraphRunContext, Edge

...

@dataclass
class Ask(BaseNode[QuestionState]):
    """Generate question using GPT-4o."""
    docstring_notes = True
    async def run(
        self, ctx: GraphRunContext[QuestionState]
    ) -> Annotated[Answer, Edge(label='Ask the question')]:
        ...

...

@dataclass
class Evaluate(BaseNode[QuestionState]):
    answer: str

    async def run(
            self,
            ctx: GraphRunContext[QuestionState],
    ) -> Annotated[End[str], Edge(label='success')] | Reprimand:
        ...

...

question_graph.mermaid_save('image.png', highlighted_nodes=[Answer])
```

*(This example is not complete and cannot be run directly)*

Would generate and image that looks like this:

### Setting Direction of the State Diagram

You can specify the direction of the state diagram using one of the following values:

- `'TB'`: Top to bottom, the diagram flows vertically from top to bottom.
- `'LR'`: Left to right, the diagram flows horizontally from left to right.
- `'RL'`: Right to left, the diagram flows horizontally from right to left.
- `'BT'`: Bottom to top, the diagram flows vertically from bottom to top.

Here is an example of how to do this using 'Left to Right' (LR) instead of the default 'Top to Bottom' (TB)

vending\_machine\_diagram.py

```py
from vending_machine import InsertCoin, vending_machine_graph

vending_machine_graph.mermaid_code(start_node=InsertCoin, direction='LR')
```

# EXAMPLES

---
title: "Pydantic Model - PydanticAI"
source: "https://ai.pydantic.dev/examples/pydantic-model/"
author:
published:
created: 2025-02-11
description: "Agent Framework / shim to use Pydantic with LLMs"
tags:
  - "clippings"
---
1. [Introduction](https://ai.pydantic.dev/)
2. [Examples](https://ai.pydantic.dev/examples/)

Version Notice

This documentation is ahead of the last release by [3 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.23...main). You may see documentation for features not yet supported in the latest release [v0.0.23 2025-02-07](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.23).

Simple example of using PydanticAI to construct a Pydantic model from a text input.

Demonstrates:

- [structured `result_type`](https://ai.pydantic.dev/results/#structured-result-validation)

## Running the Example

With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/#usage), run:

```
python -m pydantic_ai_examples.pydantic_model
```

```
uv run -m pydantic_ai_examples.pydantic_model
```

This examples uses `openai:gpt-4o` by default, but it works well with other models, e.g. you can run it with Gemini using:

```
PYDANTIC_AI_MODEL=gemini-1.5-pro python -m pydantic_ai_examples.pydantic_model
```

```
PYDANTIC_AI_MODEL=gemini-1.5-pro uv run -m pydantic_ai_examples.pydantic_model
```

(or `PYDANTIC_AI_MODEL=gemini-1.5-flash ...`)

## Example Code

pydantic\_model.py

```python
import os
from typing import cast

import logfire
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

class MyModel(BaseModel):
    city: str
    country: str

model = cast(KnownModelName, os.getenv('PYDANTIC_AI_MODEL', 'openai:gpt-4o'))
print(f'Using model: {model}')
agent = Agent(model, result_type=MyModel)

if __name__ == '__main__':
    result = agent.run_sync('The windy city in the US of A.')
    print(result.data)
    print(result.usage())
```

---
title: "Weather agent - PydanticAI"
source: "https://ai.pydantic.dev/examples/weather-agent/"
author:
published:
created: 2025-02-11
description: "Agent Framework / shim to use Pydantic with LLMs"
tags:
  - "clippings"
---
Version Notice

This documentation is ahead of the last release by [3 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.23...main). You may see documentation for features not yet supported in the latest release [v0.0.23 2025-02-07](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.23).

Example of PydanticAI with multiple tools which the LLM needs to call in turn to answer a question.

Demonstrates:

- [tools](https://ai.pydantic.dev/tools/)
- [agent dependencies](https://ai.pydantic.dev/dependencies/)
- [streaming text responses](https://ai.pydantic.dev/results/#streaming-text)
- Building a [Gradio](https://www.gradio.app/) UI for the agent

In this case the idea is a "weather" agent — the user can ask for the weather in multiple locations, the agent will use the `get_lat_lng` tool to get the latitude and longitude of the locations, then use the `get_weather` tool to get the weather for those locations.

## Running the Example

To run this example properly, you might want to add two extra API keys **(Note if either key is missing, the code will fall back to dummy data, so they're not required)**:

- A weather API key from [tomorrow.io](https://www.tomorrow.io/weather-api/) set via `WEATHER_API_KEY`
- A geocoding API key from [geocode.maps.co](https://geocode.maps.co/) set via `GEO_API_KEY`

With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/#usage), run:

```
python -m pydantic_ai_examples.weather_agent
```

```
uv run -m pydantic_ai_examples.weather_agent
```

## Example Code

pydantic\_ai\_examples/weather\_agent.py

```python
from __future__ import annotations as _annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import logfire
from devtools import debug
from httpx import AsyncClient

from pydantic_ai import Agent, ModelRetry, RunContext

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

@dataclass
class Deps:
    client: AsyncClient
    weather_api_key: str | None
    geo_api_key: str | None

weather_agent = Agent(
    'openai:gpt-4o',
    # 'Be concise, reply with one sentence.' is enough for some models (like openai) to use
    # the below tools appropriately, but others like anthropic and gemini require a bit more direction.
    system_prompt=(
        'Be concise, reply with one sentence.'
        'Use the \`get_lat_lng\` tool to get the latitude and longitude of the locations, '
        'then use the \`get_weather\` tool to get the weather.'
    ),
    deps_type=Deps,
    retries=2,
)

@weather_agent.tool
async def get_lat_lng(
    ctx: RunContext[Deps], location_description: str
) -> dict[str, float]:
    """Get the latitude and longitude of a location.

    Args:
        ctx: The context.
        location_description: A description of a location.
    """
    if ctx.deps.geo_api_key is None:
        # if no API key is provided, return a dummy response (London)
        return {'lat': 51.1, 'lng': -0.1}

    params = {
        'q': location_description,
        'api_key': ctx.deps.geo_api_key,
    }
    with logfire.span('calling geocode API', params=params) as span:
        r = await ctx.deps.client.get('https://geocode.maps.co/search', params=params)
        r.raise_for_status()
        data = r.json()
        span.set_attribute('response', data)

    if data:
        return {'lat': data[0]['lat'], 'lng': data[0]['lon']}
    else:
        raise ModelRetry('Could not find the location')

@weather_agent.tool
async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
    """Get the weather at a location.

    Args:
        ctx: The context.
        lat: Latitude of the location.
        lng: Longitude of the location.
    """
    if ctx.deps.weather_api_key is None:
        # if no API key is provided, return a dummy response
        return {'temperature': '21 °C', 'description': 'Sunny'}

    params = {
        'apikey': ctx.deps.weather_api_key,
        'location': f'{lat},{lng}',
        'units': 'metric',
    }
    with logfire.span('calling weather API', params=params) as span:
        r = await ctx.deps.client.get(
            'https://api.tomorrow.io/v4/weather/realtime', params=params
        )
        r.raise_for_status()
        data = r.json()
        span.set_attribute('response', data)

    values = data['data']['values']
    # https://docs.tomorrow.io/reference/data-layers-weather-codes
    code_lookup = {
        1000: 'Clear, Sunny',
        1100: 'Mostly Clear',
        1101: 'Partly Cloudy',
        1102: 'Mostly Cloudy',
        1001: 'Cloudy',
        2000: 'Fog',
        2100: 'Light Fog',
        4000: 'Drizzle',
        4001: 'Rain',
        4200: 'Light Rain',
        4201: 'Heavy Rain',
        5000: 'Snow',
        5001: 'Flurries',
        5100: 'Light Snow',
        5101: 'Heavy Snow',
        6000: 'Freezing Drizzle',
        6001: 'Freezing Rain',
        6200: 'Light Freezing Rain',
        6201: 'Heavy Freezing Rain',
        7000: 'Ice Pellets',
        7101: 'Heavy Ice Pellets',
        7102: 'Light Ice Pellets',
        8000: 'Thunderstorm',
    }
    return {
        'temperature': f'{values["temperatureApparent"]:0.0f}°C',
        'description': code_lookup.get(values['weatherCode'], 'Unknown'),
    }

async def main():
    async with AsyncClient() as client:
        # create a free API key at https://www.tomorrow.io/weather-api/
        weather_api_key = os.getenv('WEATHER_API_KEY')
        # create a free API key at https://geocode.maps.co/
        geo_api_key = os.getenv('GEO_API_KEY')
        deps = Deps(
            client=client, weather_api_key=weather_api_key, geo_api_key=geo_api_key
        )
        result = await weather_agent.run(
            'What is the weather like in London and in Wiltshire?', deps=deps
        )
        debug(result)
        print('Response:', result.data)

if __name__ == '__main__':
    asyncio.run(main())
```

## Running the UI

You can build multi-turn chat applications for your agent with [Gradio](https://www.gradio.app/), a framework for building AI web applications entirely in python. Gradio comes with built-in chat components and agent support so the entire UI will be implemented in a single python file!

Here's what the UI looks like for the weather agent:

Note, to run the UI, you'll need Python 3.10+.

```bash
pip install gradio>=5.9.0
python/uv-run -m pydantic_ai_examples.weather_agent_gradio
```

## UI Code

pydantic\_ai\_examples/weather\_agent\_gradio.py

```python
#! pydantic_ai_examples/weather_agent_gradio.py
```

---
title: "Bank support - PydanticAI"
source: "https://ai.pydantic.dev/examples/bank-support/"
author:
published:
created: 2025-02-11
description: "Agent Framework / shim to use Pydantic with LLMs"
tags:
  - "clippings"
---
Version Notice

This documentation is ahead of the last release by [3 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.23...main). You may see documentation for features not yet supported in the latest release [v0.0.23 2025-02-07](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.23).

Small but complete example of using PydanticAI to build a support agent for a bank.

Demonstrates:

- [dynamic system prompt](https://ai.pydantic.dev/agents/#system-prompts)
- [structured `result_type`](https://ai.pydantic.dev/results/#structured-result-validation)
- [tools](https://ai.pydantic.dev/tools/)

## Running the Example

With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/#usage), run:

```
python -m pydantic_ai_examples.bank_support
```

```
uv run -m pydantic_ai_examples.bank_support
```

(or `PYDANTIC_AI_MODEL=gemini-1.5-flash ...`)

## Example Code

bank\_support.py

```python
from dataclasses import dataclass

from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext

class DatabaseConn:
    """This is a fake database for example purposes.

    In reality, you'd be connecting to an external database
    (e.g. PostgreSQL) to get information about customers.
    """

    @classmethod
    async def customer_name(cls, *, id: int) -> str | None:
        if id == 123:
            return 'John'

    @classmethod
    async def customer_balance(cls, *, id: int, include_pending: bool) -> float:
        if id == 123 and include_pending:
            return 123.45
        else:
            raise ValueError('Customer not found')

@dataclass
class SupportDependencies:
    customer_id: int
    db: DatabaseConn

class SupportResult(BaseModel):
    support_advice: str = Field(description='Advice returned to the customer')
    block_card: bool = Field(description='Whether to block their card or not')
    risk: int = Field(description='Risk level of query', ge=0, le=10)

support_agent = Agent(
    'openai:gpt-4o',
    deps_type=SupportDependencies,
    result_type=SupportResult,
    system_prompt=(
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query. '
        "Reply using the customer's name."
    ),
)

@support_agent.system_prompt
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"

@support_agent.tool
async def customer_balance(
    ctx: RunContext[SupportDependencies], include_pending: bool
) -> str:
    """Returns the customer's current account balance."""
    balance = await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )
    return f'${balance:.2f}'

if __name__ == '__main__':
    deps = SupportDependencies(customer_id=123, db=DatabaseConn())
    result = support_agent.run_sync('What is my balance?', deps=deps)
    print(result.data)
    """
    support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
    """

    result = support_agent.run_sync('I just lost my card!', deps=deps)
    print(result.data)
    """
    support_advice="I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions." block_card=True risk=8
    """
```

---
title: "SQL Generation - PydanticAI"
source: "https://ai.pydantic.dev/examples/sql-gen/"
author:
published:
created: 2025-02-11
description: "Agent Framework / shim to use Pydantic with LLMs"
tags:
  - "clippings"
---
Version Notice

This documentation is ahead of the last release by [3 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.23...main). You may see documentation for features not yet supported in the latest release [v0.0.23 2025-02-07](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.23).

Example demonstrating how to use PydanticAI to generate SQL queries based on user input.

Demonstrates:

- [dynamic system prompt](https://ai.pydantic.dev/agents/#system-prompts)
- [structured `result_type`](https://ai.pydantic.dev/results/#structured-result-validation)
- [result validation](https://ai.pydantic.dev/results/#result-validators-functions)
- [agent dependencies](https://ai.pydantic.dev/dependencies/)

## Running the Example

The resulting SQL is validated by running it as an `EXPLAIN` query on PostgreSQL. To run the example, you first need to run PostgreSQL, e.g. via Docker:

```bash
docker run --rm -e POSTGRES_PASSWORD=postgres -p 54320:5432 postgres
```

*(we run postgres on port `54320` to avoid conflicts with any other postgres instances you may have running)*

With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/#usage), run:

```
python -m pydantic_ai_examples.sql_gen
```

```
uv run -m pydantic_ai_examples.sql_gen
```

or to use a custom prompt:

```
python -m pydantic_ai_examples.sql_gen "find me errors"
```

```
uv run -m pydantic_ai_examples.sql_gen "find me errors"
```

This model uses `gemini-1.5-flash` by default since Gemini is good at single shot queries of this kind.

## Example Code

sql\_gen.py

```python
import asyncio
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, Union

import asyncpg
import logfire
from annotated_types import MinLen
from devtools import debug
from pydantic import BaseModel, Field
from typing_extensions import TypeAlias

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.format_as_xml import format_as_xml

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_asyncpg()

DB_SCHEMA = """
CREATE TABLE records (
    created_at timestamptz,
    start_timestamp timestamptz,
    end_timestamp timestamptz,
    trace_id text,
    span_id text,
    parent_span_id text,
    level log_level,
    span_name text,
    message text,
    attributes_json_schema text,
    attributes jsonb,
    tags text[],
    is_exception boolean,
    otel_status_message text,
    service_name text
);
"""
SQL_EXAMPLES = [
    {
        'request': 'show me records where foobar is false',
        'response': "SELECT * FROM records WHERE attributes->>'foobar' = false",
    },
    {
        'request': 'show me records where attributes include the key "foobar"',
        'response': "SELECT * FROM records WHERE attributes ? 'foobar'",
    },
    {
        'request': 'show me records from yesterday',
        'response': "SELECT * FROM records WHERE start_timestamp::date > CURRENT_TIMESTAMP - INTERVAL '1 day'",
    },
    {
        'request': 'show me error records with the tag "foobar"',
        'response': "SELECT * FROM records WHERE level = 'error' and 'foobar' = ANY(tags)",
    },
]

@dataclass
class Deps:
    conn: asyncpg.Connection

class Success(BaseModel):
    """Response when SQL could be successfully generated."""

    sql_query: Annotated[str, MinLen(1)]
    explanation: str = Field(
        '', description='Explanation of the SQL query, as markdown'
    )

class InvalidRequest(BaseModel):
    """Response the user input didn't include enough information to generate SQL."""

    error_message: str

Response: TypeAlias = Union[Success, InvalidRequest]
agent: Agent[Deps, Response] = Agent(
    'google-gla:gemini-1.5-flash',
    # Type ignore while we wait for PEP-0747, nonetheless unions will work fine everywhere else
    result_type=Response,  # type: ignore
    deps_type=Deps,
)

@agent.system_prompt
async def system_prompt() -> str:
    return f"""\
Given the following PostgreSQL table of records, your job is to
write a SQL query that suits the user's request.

Database schema:

{DB_SCHEMA}

today's date = {date.today()}

{format_as_xml(SQL_EXAMPLES)}
"""

@agent.result_validator
async def validate_result(ctx: RunContext[Deps], result: Response) -> Response:
    if isinstance(result, InvalidRequest):
        return result

    # gemini often adds extraneous backslashes to SQL
    result.sql_query = result.sql_query.replace('\\', '')
    if not result.sql_query.upper().startswith('SELECT'):
        raise ModelRetry('Please create a SELECT query')

    try:
        await ctx.deps.conn.execute(f'EXPLAIN {result.sql_query}')
    except asyncpg.exceptions.PostgresError as e:
        raise ModelRetry(f'Invalid query: {e}') from e
    else:
        return result

async def main():
    if len(sys.argv) == 1:
        prompt = 'show me logs from yesterday, with level "error"'
    else:
        prompt = sys.argv[1]

    async with database_connect(
        'postgresql://postgres:postgres@localhost:54320', 'pydantic_ai_sql_gen'
    ) as conn:
        deps = Deps(conn)
        result = await agent.run(prompt, deps=deps)
    debug(result.data)

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
@asynccontextmanager
async def database_connect(server_dsn: str, database: str) -> AsyncGenerator[Any, None]:
    with logfire.span('check and create DB'):
        conn = await asyncpg.connect(server_dsn)
        try:
            db_exists = await conn.fetchval(
                'SELECT 1 FROM pg_database WHERE datname = $1', database
            )
            if not db_exists:
                await conn.execute(f'CREATE DATABASE {database}')
        finally:
            await conn.close()

    conn = await asyncpg.connect(f'{server_dsn}/{database}')
    try:
        with logfire.span('create schema'):
            async with conn.transaction():
                if not db_exists:
                    await conn.execute(
                        "CREATE TYPE log_level AS ENUM ('debug', 'info', 'warning', 'error', 'critical')"
                    )
                await conn.execute(DB_SCHEMA)
        yield conn
    finally:
        await conn.close()

if __name__ == '__main__':
    asyncio.run(main())
```

---
title: "Flight booking - PydanticAI"
source: "https://ai.pydantic.dev/examples/flight-booking/"
author:
published:
created: 2025-02-11
description: "Agent Framework / shim to use Pydantic with LLMs"
tags:
  - "clippings"
---
Example of a multi-agent flow where one agent delegates work to another, then hands off control to a third agent.

In this scenario, a group of agents work together to find the best flight for a user.

flight\_booking.py

```python
import datetime
from dataclasses import dataclass
from typing import Literal

import logfire
from pydantic import BaseModel, Field
from rich.prompt import Prompt

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import Usage, UsageLimits

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

class FlightDetails(BaseModel):
    """Details of the most suitable flight."""

    flight_number: str
    price: int
    origin: str = Field(description='Three-letter airport code')
    destination: str = Field(description='Three-letter airport code')
    date: datetime.date

class NoFlightFound(BaseModel):
    """When no valid flight is found."""

@dataclass
class Deps:
    web_page_text: str
    req_origin: str
    req_destination: str
    req_date: datetime.date

# This agent is responsible for controlling the flow of the conversation.
search_agent = Agent[Deps, FlightDetails | NoFlightFound](
    'openai:gpt-4o',
    result_type=FlightDetails | NoFlightFound,  # type: ignore
    retries=4,
    system_prompt=(
        'Your job is to find the cheapest flight for the user on the given date. '
    ),
)

# This agent is responsible for extracting flight details from web page text.
extraction_agent = Agent(
    'openai:gpt-4o',
    result_type=list[FlightDetails],
    system_prompt='Extract all the flight details from the given text.',
)

@search_agent.tool
async def extract_flights(ctx: RunContext[Deps]) -> list[FlightDetails]:
    """Get details of all flights."""
    # we pass the usage to the search agent so requests within this agent are counted
    result = await extraction_agent.run(ctx.deps.web_page_text, usage=ctx.usage)
    logfire.info('found {flight_count} flights', flight_count=len(result.data))
    return result.data

@search_agent.result_validator
async def validate_result(
    ctx: RunContext[Deps], result: FlightDetails | NoFlightFound
) -> FlightDetails | NoFlightFound:
    """Procedural validation that the flight meets the constraints."""
    if isinstance(result, NoFlightFound):
        return result

    errors: list[str] = []
    if result.origin != ctx.deps.req_origin:
        errors.append(
            f'Flight should have origin {ctx.deps.req_origin}, not {result.origin}'
        )
    if result.destination != ctx.deps.req_destination:
        errors.append(
            f'Flight should have destination {ctx.deps.req_destination}, not {result.destination}'
        )
    if result.date != ctx.deps.req_date:
        errors.append(f'Flight should be on {ctx.deps.req_date}, not {result.date}')

    if errors:
        raise ModelRetry('\n'.join(errors))
    else:
        return result

class SeatPreference(BaseModel):
    row: int = Field(ge=1, le=30)
    seat: Literal['A', 'B', 'C', 'D', 'E', 'F']

class Failed(BaseModel):
    """Unable to extract a seat selection."""

# This agent is responsible for extracting the user's seat selection
seat_preference_agent = Agent[
    None, SeatPreference | Failed
](
    'openai:gpt-4o',
    result_type=SeatPreference | Failed,  # type: ignore
    system_prompt=(
        "Extract the user's seat preference. "
        'Seats A and F are window seats. '
        'Row 1 is the front row and has extra leg room. '
        'Rows 14, and 20 also have extra leg room. '
    ),
)

# in reality this would be downloaded from a booking site,
# potentially using another agent to navigate the site
flights_web_page = """
1. Flight SFO-AK123
- Price: $350
- Origin: San Francisco International Airport (SFO)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 10, 2025

2. Flight SFO-AK456
- Price: $370
- Origin: San Francisco International Airport (SFO)
- Destination: Fairbanks International Airport (FAI)
- Date: January 10, 2025

3. Flight SFO-AK789
- Price: $400
- Origin: San Francisco International Airport (SFO)
- Destination: Juneau International Airport (JNU)
- Date: January 20, 2025

4. Flight NYC-LA101
- Price: $250
- Origin: San Francisco International Airport (SFO)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 10, 2025

5. Flight CHI-MIA202
- Price: $200
- Origin: Chicago O'Hare International Airport (ORD)
- Destination: Miami International Airport (MIA)
- Date: January 12, 2025

6. Flight BOS-SEA303
- Price: $120
- Origin: Boston Logan International Airport (BOS)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 12, 2025

7. Flight DFW-DEN404
- Price: $150
- Origin: Dallas/Fort Worth International Airport (DFW)
- Destination: Denver International Airport (DEN)
- Date: January 10, 2025

8. Flight ATL-HOU505
- Price: $180
- Origin: Hartsfield-Jackson Atlanta International Airport (ATL)
- Destination: George Bush Intercontinental Airport (IAH)
- Date: January 10, 2025
"""

# restrict how many requests this app can make to the LLM
usage_limits = UsageLimits(request_limit=15)

async def main():
    deps = Deps(
        web_page_text=flights_web_page,
        req_origin='SFO',
        req_destination='ANC',
        req_date=datetime.date(2025, 1, 10),
    )
    message_history: list[ModelMessage] | None = None
    usage: Usage = Usage()
    # run the agent until a satisfactory flight is found
    while True:
        result = await search_agent.run(
            f'Find me a flight from {deps.req_origin} to {deps.req_destination} on {deps.req_date}',
            deps=deps,
            usage=usage,
            message_history=message_history,
            usage_limits=usage_limits,
        )
        if isinstance(result.data, NoFlightFound):
            print('No flight found')
            break
        else:
            flight = result.data
            print(f'Flight found: {flight}')
            answer = Prompt.ask(
                'Do you want to buy this flight, or keep searching? (buy/*search)',
                choices=['buy', 'search', ''],
                show_choices=False,
            )
            if answer == 'buy':
                seat = await find_seat(usage)
                await buy_tickets(flight, seat)
                break
            else:
                message_history = result.all_messages(
                    result_tool_return_content='Please suggest another flight'
                )

async def find_seat(usage: Usage) -> SeatPreference:
    message_history: list[ModelMessage] | None = None
    while True:
        answer = Prompt.ask('What seat would you like?')

        result = await seat_preference_agent.run(
            answer,
            message_history=message_history,
            usage=usage,
            usage_limits=usage_limits,
        )
        if isinstance(result.data, SeatPreference):
            return result.data
        else:
            print('Could not understand seat preference. Please try again.')
            message_history = result.all_messages()

async def buy_tickets(flight_details: FlightDetails, seat: SeatPreference):
    print(f'Purchasing flight {flight_details=!r} {seat=!r}...')

if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
```

---
title: "RAG - PydanticAI"
source: "https://ai.pydantic.dev/examples/rag/"
author:
published:
created: 2025-02-11
description: "Agent Framework / shim to use Pydantic with LLMs"
tags:
  - "clippings"
---
Version Notice

This documentation is ahead of the last release by [3 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.23...main). You may see documentation for features not yet supported in the latest release [v0.0.23 2025-02-07](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.23).

RAG search example. This demo allows you to ask question of the [logfire](https://pydantic.dev/logfire) documentation.

Demonstrates:

- [tools](https://ai.pydantic.dev/tools/)
- [agent dependencies](https://ai.pydantic.dev/dependencies/)
- RAG search

This is done by creating a database containing each section of the markdown documentation, then registering the search tool with the PydanticAI agent.

Logic for extracting sections from markdown files and a JSON file with that data is available in [this gist](https://gist.github.com/samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992).

[PostgreSQL with pgvector](https://github.com/pgvector/pgvector) is used as the search database, the easiest way to download and run pgvector is using Docker:

```bash
mkdir postgres-data
docker run --rm \
  -e POSTGRES_PASSWORD=postgres \
  -p 54320:5432 \
  -v \`pwd\`/postgres-data:/var/lib/postgresql/data \
  pgvector/pgvector:pg17
```

As with the [SQL gen](https://ai.pydantic.dev/examples/sql-gen/) example, we run postgres on port `54320` to avoid conflicts with any other postgres instances you may have running. We also mount the PostgreSQL `data` directory locally to persist the data if you need to stop and restart the container.

With that running and [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/#usage), we can build the search database with (**WARNING**: this requires the `OPENAI_API_KEY` env variable and will calling the OpenAI embedding API around 300 times to generate embeddings for each section of the documentation):

```
python -m pydantic_ai_examples.rag build
```

```
uv run -m pydantic_ai_examples.rag build
```

(Note building the database doesn't use PydanticAI right now, instead it uses the OpenAI SDK directly.)

You can then ask the agent a question with:

```
python -m pydantic_ai_examples.rag search "How do I configure logfire to work with FastAPI?"
```

```
uv run -m pydantic_ai_examples.rag search "How do I configure logfire to work with FastAPI?"
```

## Example Code

rag.py

```python
from __future__ import annotations as _annotations

import asyncio
import re
import sys
import unicodedata
from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncpg
import httpx
import logfire
import pydantic_core
from openai import AsyncOpenAI
from pydantic import TypeAdapter
from typing_extensions import AsyncGenerator

from pydantic_ai import RunContext
from pydantic_ai.agent import Agent

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_asyncpg()

@dataclass
class Deps:
    openai: AsyncOpenAI
    pool: asyncpg.Pool

agent = Agent('openai:gpt-4o', deps_type=Deps)

@agent.tool
async def retrieve(context: RunContext[Deps], search_query: str) -> str:
    """Retrieve documentation sections based on a search query.

    Args:
        context: The call context.
        search_query: The search query.
    """
    with logfire.span(
        'create embedding for {search_query=}', search_query=search_query
    ):
        embedding = await context.deps.openai.embeddings.create(
            input=search_query,
            model='text-embedding-3-small',
        )

    assert (
        len(embedding.data) == 1
    ), f'Expected 1 embedding, got {len(embedding.data)}, doc query: {search_query!r}'
    embedding = embedding.data[0].embedding
    embedding_json = pydantic_core.to_json(embedding).decode()
    rows = await context.deps.pool.fetch(
        'SELECT url, title, content FROM doc_sections ORDER BY embedding <-> $1 LIMIT 8',
        embedding_json,
    )
    return '\n\n'.join(
        f'# {row["title"]}\nDocumentation URL:{row["url"]}\n\n{row["content"]}\n'
        for row in rows
    )

async def run_agent(question: str):
    """Entry point to run the agent and perform RAG based question answering."""
    openai = AsyncOpenAI()
    logfire.instrument_openai(openai)

    logfire.info('Asking "{question}"', question=question)

    async with database_connect(False) as pool:
        deps = Deps(openai=openai, pool=pool)
        answer = await agent.run(question, deps=deps)
    print(answer.data)

#######################################################
# The rest of this file is dedicated to preparing the #
# search database, and some utilities.                #
#######################################################

# JSON document from
# https://gist.github.com/samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992
DOCS_JSON = (
    'https://gist.githubusercontent.com/'
    'samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992/raw/'
    '80c5925c42f1442c24963aaf5eb1a324d47afe95/logfire_docs.json'
)

async def build_search_db():
    """Build the search database."""
    async with httpx.AsyncClient() as client:
        response = await client.get(DOCS_JSON)
        response.raise_for_status()
    sections = sessions_ta.validate_json(response.content)

    openai = AsyncOpenAI()
    logfire.instrument_openai(openai)

    async with database_connect(True) as pool:
        with logfire.span('create schema'):
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(DB_SCHEMA)

        sem = asyncio.Semaphore(10)
        async with asyncio.TaskGroup() as tg:
            for section in sections:
                tg.create_task(insert_doc_section(sem, openai, pool, section))

async def insert_doc_section(
    sem: asyncio.Semaphore,
    openai: AsyncOpenAI,
    pool: asyncpg.Pool,
    section: DocsSection,
) -> None:
    async with sem:
        url = section.url()
        exists = await pool.fetchval('SELECT 1 FROM doc_sections WHERE url = $1', url)
        if exists:
            logfire.info('Skipping {url=}', url=url)
            return

        with logfire.span('create embedding for {url=}', url=url):
            embedding = await openai.embeddings.create(
                input=section.embedding_content(),
                model='text-embedding-3-small',
            )
        assert (
            len(embedding.data) == 1
        ), f'Expected 1 embedding, got {len(embedding.data)}, doc section: {section}'
        embedding = embedding.data[0].embedding
        embedding_json = pydantic_core.to_json(embedding).decode()
        await pool.execute(
            'INSERT INTO doc_sections (url, title, content, embedding) VALUES ($1, $2, $3, $4)',
            url,
            section.title,
            section.content,
            embedding_json,
        )

@dataclass
class DocsSection:
    id: int
    parent: int | None
    path: str
    level: int
    title: str
    content: str

    def url(self) -> str:
        url_path = re.sub(r'\.md$', '', self.path)
        return (
            f'https://logfire.pydantic.dev/docs/{url_path}/#{slugify(self.title, "-")}'
        )

    def embedding_content(self) -> str:
        return '\n\n'.join((f'path: {self.path}', f'title: {self.title}', self.content))

sessions_ta = TypeAdapter(list[DocsSection])

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
@asynccontextmanager
async def database_connect(
    create_db: bool = False,
) -> AsyncGenerator[asyncpg.Pool, None]:
    server_dsn, database = (
        'postgresql://postgres:postgres@localhost:54320',
        'pydantic_ai_rag',
    )
    if create_db:
        with logfire.span('check and create DB'):
            conn = await asyncpg.connect(server_dsn)
            try:
                db_exists = await conn.fetchval(
                    'SELECT 1 FROM pg_database WHERE datname = $1', database
                )
                if not db_exists:
                    await conn.execute(f'CREATE DATABASE {database}')
            finally:
                await conn.close()

    pool = await asyncpg.create_pool(f'{server_dsn}/{database}')
    try:
        yield pool
    finally:
        await pool.close()

DB_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS doc_sections (
    id serial PRIMARY KEY,
    url text NOT NULL UNIQUE,
    title text NOT NULL,
    content text NOT NULL,
    -- text-embedding-3-small returns a vector of 1536 floats
    embedding vector(1536) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_sections_embedding ON doc_sections USING hnsw (embedding vector_l2_ops);
"""

def slugify(value: str, separator: str, unicode: bool = False) -> str:
    """Slugify a string, to make it URL friendly."""
    # Taken unchanged from https://github.com/Python-Markdown/markdown/blob/3.7/markdown/extensions/toc.py#L38
    if not unicode:
        # Replace Extended Latin characters with ASCII, i.e. \`žlutý\` => \`zluty\`
        value = unicodedata.normalize('NFKD', value)
        value = value.encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(rf'[{separator}\s]+', separator, value)

if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else None
    if action == 'build':
        asyncio.run(build_search_db())
    elif action == 'search':
        if len(sys.argv) == 3:
            q = sys.argv[2]
        else:
            q = 'How do I configure logfire to work with FastAPI?'
        asyncio.run(run_agent(q))
    else:
        print(
            'uv run --extra examples -m pydantic_ai_examples.rag build|search',
            file=sys.stderr,
        )
        sys.exit(1)
```

---
title: "Stream markdown - PydanticAI"
source: "https://ai.pydantic.dev/examples/stream-markdown/"
author:
published:
created: 2025-02-11
description: "Agent Framework / shim to use Pydantic with LLMs"
tags:
  - "clippings"
---
Version Notice

This documentation is ahead of the last release by [3 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.23...main). You may see documentation for features not yet supported in the latest release [v0.0.23 2025-02-07](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.23).

This example shows how to stream markdown from an agent, using the [`rich`](https://github.com/Textualize/rich) library to highlight the output in the terminal.

It'll run the example with both OpenAI and Google Gemini models if the required environment variables are set.

Demonstrates:

- [streaming text responses](https://ai.pydantic.dev/results/#streaming-text)

## Running the Example

With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/#usage), run:

```
python -m pydantic_ai_examples.stream_markdown
```

```
uv run -m pydantic_ai_examples.stream_markdown
```

## Example Code

```python
import asyncio
import os

import logfire
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock, Markdown
from rich.syntax import Syntax
from rich.text import Text

from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

agent = Agent()

# models to try, and the appropriate env var
models: list[tuple[KnownModelName, str]] = [
    ('google-gla:gemini-1.5-flash', 'GEMINI_API_KEY'),
    ('openai:gpt-4o-mini', 'OPENAI_API_KEY'),
    ('groq:llama-3.3-70b-versatile', 'GROQ_API_KEY'),
]

async def main():
    prettier_code_blocks()
    console = Console()
    prompt = 'Show me a short example of using Pydantic.'
    console.log(f'Asking: {prompt}...', style='cyan')
    for model, env_var in models:
        if env_var in os.environ:
            console.log(f'Using model: {model}')
            with Live('', console=console, vertical_overflow='visible') as live:
                async with agent.run_stream(prompt, model=model) as result:
                    async for message in result.stream():
                        live.update(Markdown(message))
            console.log(result.usage())
        else:
            console.log(f'{model} requires {env_var} to be set.')

def prettier_code_blocks():
    """Make rich code blocks prettier and easier to copy.

    From https://github.com/samuelcolvin/aicli/blob/v0.8.0/samuelcolvin_aicli.py#L22
    """

    class SimpleCodeBlock(CodeBlock):
        def __rich_console__(
            self, console: Console, options: ConsoleOptions
        ) -> RenderResult:
            code = str(self.text).rstrip()
            yield Text(self.lexer_name, style='dim')
            yield Syntax(
                code,
                self.lexer_name,
                theme=self.theme,
                background_color='default',
                word_wrap=True,
            )
            yield Text(f'/{self.lexer_name}', style='dim')

    Markdown.elements['fence'] = SimpleCodeBlock

if __name__ == '__main__':
    asyncio.run(main())
```

---
title: "Stream whales - PydanticAI"
source: "https://ai.pydantic.dev/examples/stream-whales/"
author:
published:
created: 2025-02-11
description: "Agent Framework / shim to use Pydantic with LLMs"
tags:
  - "clippings"
---
Version Notice

This documentation is ahead of the last release by [3 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.23...main). You may see documentation for features not yet supported in the latest release [v0.0.23 2025-02-07](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.23).

Information about whales — an example of streamed structured response validation.

Demonstrates:

- [streaming structured responses](https://ai.pydantic.dev/results/#streaming-structured-responses)

This script streams structured responses from GPT-4 about whales, validates the data and displays it as a dynamic table using [`rich`](https://github.com/Textualize/rich) as the data is received.

## Running the Example

With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/#usage), run:

```
python -m pydantic_ai_examples.stream_whales
```

```
uv run -m pydantic_ai_examples.stream_whales
```

Should give an output like this:

## Example Code

stream\_whales.py

```python
from typing import Annotated

import logfire
from pydantic import Field, ValidationError
from rich.console import Console
from rich.live import Live
from rich.table import Table
from typing_extensions import NotRequired, TypedDict

from pydantic_ai import Agent

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

class Whale(TypedDict):
    name: str
    length: Annotated[
        float, Field(description='Average length of an adult whale in meters.')
    ]
    weight: NotRequired[
        Annotated[
            float,
            Field(description='Average weight of an adult whale in kilograms.', ge=50),
        ]
    ]
    ocean: NotRequired[str]
    description: NotRequired[Annotated[str, Field(description='Short Description')]]

agent = Agent('openai:gpt-4', result_type=list[Whale])

async def main():
    console = Console()
    with Live('\n' * 36, console=console) as live:
        console.print('Requesting data...', style='cyan')
        async with agent.run_stream(
            'Generate me details of 5 species of Whale.'
        ) as result:
            console.print('Response:', style='green')

            async for message, last in result.stream_structured(debounce_by=0.01):
                try:
                    whales = await result.validate_structured_result(
                        message, allow_partial=not last
                    )
                except ValidationError as exc:
                    if all(
                        e['type'] == 'missing' and e['loc'] == ('response',)
                        for e in exc.errors()
                    ):
                        continue
                    else:
                        raise

                table = Table(
                    title='Species of Whale',
                    caption='Streaming Structured responses from GPT-4',
                    width=120,
                )
                table.add_column('ID', justify='right')
                table.add_column('Name')
                table.add_column('Avg. Length (m)', justify='right')
                table.add_column('Avg. Weight (kg)', justify='right')
                table.add_column('Ocean')
                table.add_column('Description', justify='right')

                for wid, whale in enumerate(whales, start=1):
                    table.add_row(
                        str(wid),
                        whale['name'],
                        f'{whale["length"]:0.0f}',
                        f'{w:0.0f}' if (w := whale.get('weight')) else '…',
                        whale.get('ocean') or '…',
                        whale.get('description') or '…',
                    )
                live.update(table)

if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
```

---
title: "Chat App with FastAPI - PydanticAI"
source: "https://ai.pydantic.dev/examples/chat-app/"
author:
published:
created: 2025-02-11
description: "Agent Framework / shim to use Pydantic with LLMs"
tags:
  - "clippings"
---
Version Notice

This documentation is ahead of the last release by [3 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.23...main). You may see documentation for features not yet supported in the latest release [v0.0.23 2025-02-07](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.23).

Simple chat app example build with FastAPI.

Demonstrates:

- [reusing chat history](https://ai.pydantic.dev/message-history/)
- [serializing messages](https://ai.pydantic.dev/message-history/#accessing-messages-from-results)
- [streaming responses](https://ai.pydantic.dev/results/#streamed-results)

This demonstrates storing chat history between requests and using it to give the model context for new responses.

Most of the complex logic here is between `chat_app.py` which streams the response to the browser, and `chat_app.ts` which renders messages in the browser.

## Running the Example

With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/#usage), run:

```
python -m pydantic_ai_examples.chat_app
```

```
uv run -m pydantic_ai_examples.chat_app
```

Then open the app at [localhost:8000](http://localhost:8000/).

[![Example conversation](https://ai.pydantic.dev/img/chat-app-example.png)](https://ai.pydantic.dev/img/chat-app-example.png)

## Example Code

Python code that runs the chat app:

chat\_app.py

```python
from __future__ import annotations as _annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, TypeVar

import fastapi
import logfire
from fastapi import Depends, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from typing_extensions import LiteralString, ParamSpec, TypedDict

from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

agent = Agent('openai:gpt-4o')
THIS_DIR = Path(__file__).parent

@asynccontextmanager
async def lifespan(_app: fastapi.FastAPI):
    async with Database.connect() as db:
        yield {'db': db}

app = fastapi.FastAPI(lifespan=lifespan)
logfire.instrument_fastapi(app)

@app.get('/')
async def index() -> FileResponse:
    return FileResponse((THIS_DIR / 'chat_app.html'), media_type='text/html')

@app.get('/chat_app.ts')
async def main_ts() -> FileResponse:
    """Get the raw typescript code, it's compiled in the browser, forgive me."""
    return FileResponse((THIS_DIR / 'chat_app.ts'), media_type='text/plain')

async def get_db(request: Request) -> Database:
    return request.state.db

@app.get('/chat/')
async def get_chat(database: Database = Depends(get_db)) -> Response:
    msgs = await database.get_messages()
    return Response(
        b'\n'.join(json.dumps(to_chat_message(m)).encode('utf-8') for m in msgs),
        media_type='text/plain',
    )

class ChatMessage(TypedDict):
    """Format of messages sent to the browser."""

    role: Literal['user', 'model']
    timestamp: str
    content: str

def to_chat_message(m: ModelMessage) -> ChatMessage:
    first_part = m.parts[0]
    if isinstance(m, ModelRequest):
        if isinstance(first_part, UserPromptPart):
            return {
                'role': 'user',
                'timestamp': first_part.timestamp.isoformat(),
                'content': first_part.content,
            }
    elif isinstance(m, ModelResponse):
        if isinstance(first_part, TextPart):
            return {
                'role': 'model',
                'timestamp': m.timestamp.isoformat(),
                'content': first_part.content,
            }
    raise UnexpectedModelBehavior(f'Unexpected message type for chat app: {m}')

@app.post('/chat/')
async def post_chat(
    prompt: Annotated[str, fastapi.Form()], database: Database = Depends(get_db)
) -> StreamingResponse:
    async def stream_messages():
        """Streams new line delimited JSON \`Message\`s to the client."""
        # stream the user prompt so that can be displayed straight away
        yield (
            json.dumps(
                {
                    'role': 'user',
                    'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                    'content': prompt,
                }
            ).encode('utf-8')
            + b'\n'
        )
        # get the chat history so far to pass as context to the agent
        messages = await database.get_messages()
        # run the agent with the user prompt and the chat history
        async with agent.run_stream(prompt, message_history=messages) as result:
            async for text in result.stream(debounce_by=0.01):
                # text here is a \`str\` and the frontend wants
                # JSON encoded ModelResponse, so we create one
                m = ModelResponse(parts=[TextPart(text)], timestamp=result.timestamp())
                yield json.dumps(to_chat_message(m)).encode('utf-8') + b'\n'

        # add new messages (e.g. the user prompt and the agent response in this case) to the database
        await database.add_messages(result.new_messages_json())

    return StreamingResponse(stream_messages(), media_type='text/plain')

P = ParamSpec('P')
R = TypeVar('R')

@dataclass
class Database:
    """Rudimentary database to store chat messages in SQLite.

    The SQLite standard library package is synchronous, so we
    use a thread pool executor to run queries asynchronously.
    """

    con: sqlite3.Connection
    _loop: asyncio.AbstractEventLoop
    _executor: ThreadPoolExecutor

    @classmethod
    @asynccontextmanager
    async def connect(
        cls, file: Path = THIS_DIR / '.chat_app_messages.sqlite'
    ) -> AsyncIterator[Database]:
        with logfire.span('connect to DB'):
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            con = await loop.run_in_executor(executor, cls._connect, file)
            slf = cls(con, loop, executor)
        try:
            yield slf
        finally:
            await slf._asyncify(con.close)

    @staticmethod
    def _connect(file: Path) -> sqlite3.Connection:
        con = sqlite3.connect(str(file))
        con = logfire.instrument_sqlite3(con)
        cur = con.cursor()
        cur.execute(
            'CREATE TABLE IF NOT EXISTS messages (id INT PRIMARY KEY, message_list TEXT);'
        )
        con.commit()
        return con

    async def add_messages(self, messages: bytes):
        await self._asyncify(
            self._execute,
            'INSERT INTO messages (message_list) VALUES (?);',
            messages,
            commit=True,
        )
        await self._asyncify(self.con.commit)

    async def get_messages(self) -> list[ModelMessage]:
        c = await self._asyncify(
            self._execute, 'SELECT message_list FROM messages order by id'
        )
        rows = await self._asyncify(c.fetchall)
        messages: list[ModelMessage] = []
        for row in rows:
            messages.extend(ModelMessagesTypeAdapter.validate_json(row[0]))
        return messages

    def _execute(
        self, sql: LiteralString, *args: Any, commit: bool = False
    ) -> sqlite3.Cursor:
        cur = self.con.cursor()
        cur.execute(sql, args)
        if commit:
            self.con.commit()
        return cur

    async def _asyncify(
        self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        return await self._loop.run_in_executor(  # type: ignore
            self._executor,
            partial(func, **kwargs),
            *args,  # type: ignore
        )

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        'pydantic_ai_examples.chat_app:app', reload=True, reload_dirs=[str(THIS_DIR)]
    )
```

Simple HTML page to render the app:

chat\_app.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Chat App</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    main {
      max-width: 700px;
    }
    #conversation .user::before {
      content: 'You asked: ';
      font-weight: bold;
      display: block;
    }
    #conversation .model::before {
      content: 'AI Response: ';
      font-weight: bold;
      display: block;
    }
    #spinner {
      opacity: 0;
      transition: opacity 500ms ease-in;
      width: 30px;
      height: 30px;
      border: 3px solid #222;
      border-bottom-color: transparent;
      border-radius: 50%;
      animation: rotation 1s linear infinite;
    }
    @keyframes rotation {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    #spinner.active {
      opacity: 1;
    }
  </style>
</head>
<body>
  <main class="border rounded mx-auto my-5 p-4">
    <h1>Chat App</h1>
    <p>Ask me anything...</p>
    <div id="conversation" class="px-2"></div>
    <div class="d-flex justify-content-center mb-3">
      <div id="spinner"></div>
    </div>
    <form method="post">
      <input id="prompt-input" name="prompt" class="form-control"/>
      <div class="d-flex justify-content-end">
        <button class="btn btn-primary mt-2">Send</button>
      </div>
    </form>
    <div id="error" class="d-none text-danger">
      Error occurred, check the browser developer console for more information.
    </div>
  </main>
</body>
</html>
<script src="https://cdnjs.cloudflare.com/ajax/libs/typescript/5.6.3/typescript.min.js" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
<script type="module">
  // to let me write TypeScript, without adding the burden of npm we do a dirty, non-production-ready hack
  // and transpile the TypeScript code in the browser
  // this is (arguably) A neat demo trick, but not suitable for production!
  async function loadTs() {
    const response = await fetch('/chat_app.ts');
    const tsCode = await response.text();
    const jsCode = window.ts.transpile(tsCode, { target: "es2015" });
    let script = document.createElement('script');
    script.type = 'module';
    script.text = jsCode;
    document.body.appendChild(script);
  }

  loadTs().catch((e) => {
    console.error(e);
    document.getElementById('error').classList.remove('d-none');
    document.getElementById('spinner').classList.remove('active');
  });
</script>
```

TypeScript to handle rendering the messages, to keep this simple (and at the risk of offending frontend developers) the typescript code is passed to the browser as plain text and transpiled in the browser.

chat\_app.ts

```ts
// BIG FAT WARNING: to avoid the complexity of npm, this typescript is compiled in the browser
// there's currently no static type checking

import { marked } from 'https://cdnjs.cloudflare.com/ajax/libs/marked/15.0.0/lib/marked.esm.js'
const convElement = document.getElementById('conversation')

const promptInput = document.getElementById('prompt-input') as HTMLInputElement
const spinner = document.getElementById('spinner')

// stream the response and render messages as each chunk is received
// data is sent as newline-delimited JSON
async function onFetchResponse(response: Response): Promise<void> {
  let text = ''
  let decoder = new TextDecoder()
  if (response.ok) {
    const reader = response.body.getReader()
    while (true) {
      const {done, value} = await reader.read()
      if (done) {
        break
      }
      text += decoder.decode(value)
      addMessages(text)
      spinner.classList.remove('active')
    }
    addMessages(text)
    promptInput.disabled = false
    promptInput.focus()
  } else {
    const text = await response.text()
    console.error(\`Unexpected response: ${response.status}\`, {response, text})
    throw new Error(\`Unexpected response: ${response.status}\`)
  }
}

// The format of messages, this matches pydantic-ai both for brevity and understanding
// in production, you might not want to keep this format all the way to the frontend
interface Message {
  role: string
  content: string
  timestamp: string
}

// take raw response text and render messages into the \`#conversation\` element
// Message timestamp is assumed to be a unique identifier of a message, and is used to deduplicate
// hence you can send data about the same message multiple times, and it will be updated
// instead of creating a new message elements
function addMessages(responseText: string) {
  const lines = responseText.split('\n')
  const messages: Message[] = lines.filter(line => line.length > 1).map(j => JSON.parse(j))
  for (const message of messages) {
    // we use the timestamp as a crude element id
    const {timestamp, role, content} = message
    const id = \`msg-${timestamp}\`
    let msgDiv = document.getElementById(id)
    if (!msgDiv) {
      msgDiv = document.createElement('div')
      msgDiv.id = id
      msgDiv.title = \`${role} at ${timestamp}\`
      msgDiv.classList.add('border-top', 'pt-2', role)
      convElement.appendChild(msgDiv)
    }
    msgDiv.innerHTML = marked.parse(content)
  }
  window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })
}

function onError(error: any) {
  console.error(error)
  document.getElementById('error').classList.remove('d-none')
  document.getElementById('spinner').classList.remove('active')
}

async function onSubmit(e: SubmitEvent): Promise<void> {
  e.preventDefault()
  spinner.classList.add('active')
  const body = new FormData(e.target as HTMLFormElement)

  promptInput.value = ''
  promptInput.disabled = true

  const response = await fetch('/chat/', {method: 'POST', body})
  await onFetchResponse(response)
}

// call onSubmit when the form is submitted (e.g. user clicks the send button or hits Enter)
document.querySelector('form').addEventListener('submit', (e) => onSubmit(e).catch(onError))

// load messages on page load
fetch('/chat/').then(onFetchResponse).catch(onError)
```

---
title: "Question Graph - PydanticAI"
source: "https://ai.pydantic.dev/examples/question-graph/"
author:
published:
created: 2025-02-11
description: "Agent Framework / shim to use Pydantic with LLMs"
tags:
  - "clippings"
---
Version Notice

This documentation is ahead of the last release by [3 commits](https://github.com/pydantic/pydantic-ai/compare/v0.0.23...main). You may see documentation for features not yet supported in the latest release [v0.0.23 2025-02-07](https://github.com/pydantic/pydantic-ai/releases/tag/v0.0.23).

Example of a graph for asking and evaluating questions.

Demonstrates:

- [`pydantic_graph`](https://ai.pydantic.dev/graph/)

## Running the Example

With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/#usage), run:

```
python -m pydantic_ai_examples.question_graph
```

```
uv run -m pydantic_ai_examples.question_graph
```

## Example Code

question\_graph.py

```python
from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import logfire
from devtools import debug
from pydantic_graph import BaseNode, Edge, End, Graph, GraphRunContext, HistoryStep

from pydantic_ai import Agent
from pydantic_ai.format_as_xml import format_as_xml
from pydantic_ai.messages import ModelMessage

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

ask_agent = Agent('openai:gpt-4o', result_type=str)

@dataclass
class QuestionState:
    question: str | None = None
    ask_agent_messages: list[ModelMessage] = field(default_factory=list)
    evaluate_agent_messages: list[ModelMessage] = field(default_factory=list)

@dataclass
class Ask(BaseNode[QuestionState]):
    async def run(self, ctx: GraphRunContext[QuestionState]) -> Answer:
        result = await ask_agent.run(
            'Ask a simple question with a single correct answer.',
            message_history=ctx.state.ask_agent_messages,
        )
        ctx.state.ask_agent_messages += result.all_messages()
        ctx.state.question = result.data
        return Answer()

@dataclass
class Answer(BaseNode[QuestionState]):
    answer: str | None = None

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Evaluate:
        assert self.answer is not None
        return Evaluate(self.answer)

@dataclass
class EvaluationResult:
    correct: bool
    comment: str

evaluate_agent = Agent(
    'openai:gpt-4o',
    result_type=EvaluationResult,
    system_prompt='Given a question and answer, evaluate if the answer is correct.',
)

@dataclass
class Evaluate(BaseNode[QuestionState]):
    answer: str

    async def run(
        self,
        ctx: GraphRunContext[QuestionState],
    ) -> Congratulate | Reprimand:
        assert ctx.state.question is not None
        result = await evaluate_agent.run(
            format_as_xml({'question': ctx.state.question, 'answer': self.answer}),
            message_history=ctx.state.evaluate_agent_messages,
        )
        ctx.state.evaluate_agent_messages += result.all_messages()
        if result.data.correct:
            return Congratulate(result.data.comment)
        else:
            return Reprimand(result.data.comment)

@dataclass
class Congratulate(BaseNode[QuestionState, None, None]):
    comment: str

    async def run(
        self, ctx: GraphRunContext[QuestionState]
    ) -> Annotated[End, Edge(label='success')]:
        print(f'Correct answer! {self.comment}')
        return End(None)

@dataclass
class Reprimand(BaseNode[QuestionState]):
    comment: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Ask:
        print(f'Comment: {self.comment}')
        # > Comment: Vichy is no longer the capital of France.
        ctx.state.question = None
        return Ask()

question_graph = Graph(
    nodes=(Ask, Answer, Evaluate, Congratulate, Reprimand), state_type=QuestionState
)

async def run_as_continuous():
    state = QuestionState()
    node = Ask()
    history: list[HistoryStep[QuestionState, None]] = []
    with logfire.span('run questions graph'):
        while True:
            node = await question_graph.next(node, history, state=state)
            if isinstance(node, End):
                debug([e.data_snapshot() for e in history])
                break
            elif isinstance(node, Answer):
                assert state.question
                node.answer = input(f'{state.question} ')
            # otherwise just continue

async def run_as_cli(answer: str | None):
    history_file = Path('question_graph_history.json')
    history = (
        question_graph.load_history(history_file.read_bytes())
        if history_file.exists()
        else []
    )

    if history:
        last = history[-1]
        assert last.kind == 'node', 'expected last step to be a node'
        state = last.state
        assert answer is not None, 'answer is required to continue from history'
        node = Answer(answer)
    else:
        state = QuestionState()
        node = Ask()
    debug(state, node)

    with logfire.span('run questions graph'):
        while True:
            node = await question_graph.next(node, history, state=state)
            if isinstance(node, End):
                debug([e.data_snapshot() for e in history])
                print('Finished!')
                break
            elif isinstance(node, Answer):
                print(state.question)
                break
            # otherwise just continue

    history_file.write_bytes(question_graph.dump_history(history, indent=2))

if __name__ == '__main__':
    import asyncio
    import sys

    try:
        sub_command = sys.argv[1]
        assert sub_command in ('continuous', 'cli', 'mermaid')
    except (IndexError, AssertionError):
        print(
            'Usage:\n'
            '  uv run -m pydantic_ai_examples.question_graph mermaid\n'
            'or:\n'
            '  uv run -m pydantic_ai_examples.question_graph continuous\n'
            'or:\n'
            '  uv run -m pydantic_ai_examples.question_graph cli [answer]',
            file=sys.stderr,
        )
        sys.exit(1)

    if sub_command == 'mermaid':
        print(question_graph.mermaid_code(start_node=Ask))
    elif sub_command == 'continuous':
        asyncio.run(run_as_continuous())
    else:
        a = sys.argv[2] if len(sys.argv) > 2 else None
        asyncio.run(run_as_cli(a))
```

The mermaid diagram generated in this example looks like this: