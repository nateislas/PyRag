"""Specialized content processing for different content types."""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .semantic_chunker import SemanticChunk


# Import utility function locally to avoid circular import
def parse_llm_json_response(content: str):
    """Parse LLM JSON response with markdown code block handling."""
    import json
    import re

    # Remove markdown code blocks if present
    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```\s*$", "", content)
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        # Try to extract JSON from the response
        try:
            # Look for JSON-like content
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            pass

        # If all else fails, try to parse as a simple list
        try:
            # Look for array-like content
            array_match = re.search(r"\[.*\]", content, re.DOTALL)
            if array_match:
                return json.loads(array_match.group(0))
        except:
            pass

        raise e


from ..llm.client import LLMClient
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class CodeChunk:
    """Processed code block with syntax analysis."""

    code: str
    language: str
    syntax_highlighted: str
    extracted_elements: Dict[str, Any]  # functions, classes, imports, etc.
    complexity_score: float
    dependencies: List[str]


@dataclass
class APIDocChunk:
    """Processed API documentation."""

    content: str
    api_path: Optional[str]
    function_signature: Optional[str]
    parameters: List[Dict[str, Any]]
    return_type: Optional[str]
    examples: List[str]
    deprecation_info: Optional[Dict[str, Any]]


@dataclass
class ExampleChunk:
    """Processed code example."""

    code: str
    explanation: str
    use_case: str
    prerequisites: List[str]
    expected_output: Optional[str]
    complexity_level: str


@dataclass
class TutorialChunk:
    """Processed tutorial content."""

    content: str
    step_number: Optional[int]
    prerequisites: List[str]
    expected_outcome: str
    difficulty_level: str
    time_estimate: Optional[str]


class ContentTypeProcessor:
    """Specialized processor for different content types."""

    def __init__(self, llm_client: LLMClient):
        """Initialize content type processor."""
        self.llm_client = llm_client
        self.logger = get_logger(__name__)

    async def process_code_block(
        self, code: str, language: str = "python"
    ) -> CodeChunk:
        """Process code blocks with syntax highlighting and analysis."""
        self.logger.info(f"Processing code block in {language}")

        # Basic syntax highlighting (simplified)
        syntax_highlighted = self._highlight_syntax(code, language)

        # Extract code elements
        extracted_elements = await self._extract_code_elements(code, language)

        # Calculate complexity
        complexity_score = self._calculate_code_complexity(code, language)

        # Extract dependencies
        dependencies = self._extract_dependencies(code, language)

        return CodeChunk(
            code=code,
            language=language,
            syntax_highlighted=syntax_highlighted,
            extracted_elements=extracted_elements,
            complexity_score=complexity_score,
            dependencies=dependencies,
        )

    async def process_api_documentation(self, content: str) -> APIDocChunk:
        """Process API documentation with parameter extraction."""
        self.logger.info("Processing API documentation")

        # Extract API path
        api_path = self._extract_api_path(content)

        # Extract function signature
        function_signature = self._extract_function_signature(content)

        # Extract parameters
        parameters = await self._extract_api_parameters(content)

        # Extract return type
        return_type = await self._extract_api_return_type(content)

        # Extract examples
        examples = self._extract_api_examples(content)

        # Check for deprecation
        deprecation_info = await self._check_deprecation(content)

        return APIDocChunk(
            content=content,
            api_path=api_path,
            function_signature=function_signature,
            parameters=parameters,
            return_type=return_type,
            examples=examples,
            deprecation_info=deprecation_info,
        )

    async def process_example(self, example: str) -> ExampleChunk:
        """Process code examples with context and explanations."""
        self.logger.info("Processing code example")

        # Extract explanation
        explanation = await self._extract_example_explanation(example)

        # Determine use case
        use_case = await self._determine_use_case(example)

        # Extract prerequisites
        prerequisites = await self._extract_prerequisites(example)

        # Extract expected output
        expected_output = await self._extract_expected_output(example)

        # Determine complexity
        complexity_level = await self._determine_example_complexity(example)

        return ExampleChunk(
            code=example,
            explanation=explanation,
            use_case=use_case,
            prerequisites=prerequisites,
            expected_output=expected_output,
            complexity_level=complexity_level,
        )

    async def process_tutorial(self, content: str) -> TutorialChunk:
        """Process tutorial content with step-by-step analysis."""
        self.logger.info("Processing tutorial content")

        # Extract step number
        step_number = self._extract_step_number(content)

        # Extract prerequisites
        prerequisites = await self._extract_tutorial_prerequisites(content)

        # Extract expected outcome
        expected_outcome = await self._extract_expected_outcome(content)

        # Determine difficulty
        difficulty_level = await self._determine_tutorial_difficulty(content)

        # Extract time estimate
        time_estimate = self._extract_time_estimate(content)

        return TutorialChunk(
            content=content,
            step_number=step_number,
            prerequisites=prerequisites,
            expected_outcome=expected_outcome,
            difficulty_level=difficulty_level,
            time_estimate=time_estimate,
        )

    def _highlight_syntax(self, code: str, language: str) -> str:
        """Basic syntax highlighting (simplified implementation)."""
        if language.lower() == "python":
            # Simple Python syntax highlighting
            highlighted = code

            # Highlight keywords
            keywords = [
                "def",
                "class",
                "import",
                "from",
                "as",
                "if",
                "else",
                "elif",
                "for",
                "while",
                "try",
                "except",
                "finally",
                "with",
                "return",
                "yield",
                "async",
                "await",
            ]
            for keyword in keywords:
                highlighted = re.sub(rf"\b{keyword}\b", f"**{keyword}**", highlighted)

            # Highlight strings
            highlighted = re.sub(r'"[^"]*"', r"`\g<0>`", highlighted)
            highlighted = re.sub(r"'[^']*'", r"`\g<0>`", highlighted)

            return highlighted

        return code  # Return original if no highlighting available

    async def _extract_code_elements(self, code: str, language: str) -> Dict[str, Any]:
        """Extract code elements using LLM."""
        prompt = f"""
        Analyze this {language} code and extract key elements:

        {code}

        Return a JSON object with:
        {{
            "functions": ["function1", "function2"],
            "classes": ["class1", "class2"],
            "imports": ["import1", "import2"],
            "variables": ["var1", "var2"],
            "comments": ["comment1", "comment2"]
        }}
        """

        try:
            response = await self.llm_client.client.chat.completions.create(
                model=self.llm_client.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            result = parse_llm_json_response(content)

            return result

        except Exception as e:
            self.logger.warning(f"Code element extraction failed: {e}")
            return {
                "functions": [],
                "classes": [],
                "imports": [],
                "variables": [],
                "comments": [],
            }

    def _calculate_code_complexity(self, code: str, language: str) -> float:
        """Calculate code complexity score."""
        complexity = 0.0

        # Count lines
        lines = code.split("\n")
        complexity += min(len(lines) / 50.0, 0.3)  # Max 0.3 for length

        # Count functions/classes
        function_count = len(re.findall(r"\bdef\s+\w+", code))
        class_count = len(re.findall(r"\bclass\s+\w+", code))
        complexity += min((function_count + class_count) * 0.1, 0.3)

        # Count control structures
        control_structures = len(
            re.findall(r"\b(if|else|elif|for|while|try|except|with)\b", code)
        )
        complexity += min(control_structures * 0.05, 0.2)

        # Count nested structures
        nested_level = self._calculate_nesting_level(code)
        complexity += min(nested_level * 0.1, 0.2)

        return min(complexity, 1.0)

    def _calculate_nesting_level(self, code: str) -> int:
        """Calculate maximum nesting level in code."""
        max_level = 0
        current_level = 0

        for line in code.split("\n"):
            stripped = line.strip()
            if stripped.startswith(
                ("if ", "for ", "while ", "try:", "with ", "def ", "class ")
            ):
                current_level += 1
                max_level = max(max_level, current_level)
            elif stripped.startswith(("else:", "elif ", "except", "finally:")):
                # Same level
                pass
            elif stripped and not stripped.startswith(("#", '"""', "'''")):
                # Check if this line reduces nesting
                if current_level > 0 and not line.startswith(" " * (current_level * 4)):
                    current_level = max(0, current_level - 1)

        return max_level

    def _extract_dependencies(self, code: str, language: str) -> List[str]:
        """Extract dependencies from code."""
        dependencies = []

        if language.lower() == "python":
            # Extract imports
            import_patterns = [
                r"import\s+(\w+)",
                r"from\s+(\w+)\s+import",
                r"from\s+(\w+\.\w+)\s+import",
            ]

            for pattern in import_patterns:
                matches = re.findall(pattern, code)
                dependencies.extend(matches)

        return list(set(dependencies))  # Remove duplicates

    def _extract_api_path(self, content: str) -> Optional[str]:
        """Extract API path from documentation."""
        # Look for common API path patterns
        patterns = [
            r"`([\w\.]+)`",  # Backticks
            r"``([\w\.]+)``",  # Double backticks
            r"([\w\.]+)\(",  # Function calls
            r"class\s+(\w+)",  # Class definitions
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                return matches[0]

        return None

    def _extract_function_signature(self, content: str) -> Optional[str]:
        """Extract function signature from documentation."""
        # Look for function definitions
        func_match = re.search(r"def\s+(\w+)\s*\([^)]*\)", content)
        if func_match:
            return func_match.group(0)

        # Look for signature in documentation
        sig_match = re.search(r"(\w+)\s*\([^)]*\)", content)
        if sig_match:
            return sig_match.group(0)

        return None

    async def _extract_api_parameters(self, content: str) -> List[Dict[str, Any]]:
        """Extract API parameters using LLM."""
        prompt = f"""
        Extract function parameters from this API documentation:

        {content}

        Return a JSON array of parameter objects:
        {{
            "name": "parameter_name",
            "type": "parameter_type",
            "default": "default_value",
            "description": "parameter_description",
            "required": true/false
        }}
        """

        try:
            response = await self.llm_client.client.chat.completions.create(
                model=self.llm_client.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            import json

            result = json.loads(content)

            if isinstance(result, dict) and "parameters" in result:
                return result["parameters"]
            elif isinstance(result, list):
                return result
            else:
                return []

        except Exception as e:
            self.logger.warning(f"Parameter extraction failed: {e}")
            return []

    async def _extract_api_return_type(self, content: str) -> Optional[str]:
        """Extract API return type using LLM."""
        prompt = f"""
        Extract the return type from this API documentation:

        {content}

        Return only the return type (e.g., "str", "List[int]", "None").
        If no return type is specified, return "Any".
        """

        try:
            response = await self.llm_client.client.chat.completions.create(
                model=self.llm_client.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            self.logger.warning(f"Return type extraction failed: {e}")
            return "Any"

    def _extract_api_examples(self, content: str) -> List[str]:
        """Extract API examples from documentation."""
        # Look for code blocks
        code_blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)

        # Look for inline code
        inline_code = re.findall(r"`([^`]+)`", content)

        examples = []
        examples.extend([block.strip() for block in code_blocks])
        examples.extend([code for code in inline_code if len(code) > 10])

        return examples

    async def _check_deprecation(self, content: str) -> Optional[Dict[str, Any]]:
        """Check for deprecation information."""
        deprecation_keywords = [
            "deprecated",
            "deprecation",
            "removed",
            "obsolete",
            "legacy",
        ]

        if any(keyword in content.lower() for keyword in deprecation_keywords):
            prompt = f"""
            Analyze this content for deprecation information:

            {content}

            Return a JSON object with:
            {{
                "is_deprecated": true/false,
                "deprecation_message": "message",
                "replacement": "replacement_api",
                "removal_version": "version"
            }}
            """

            try:
                response = await self.llm_client.client.chat.completions.create(
                    model=self.llm_client.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )

                content = response.choices[0].message.content
                result = parse_llm_json_response(content)

                return result

            except Exception as e:
                self.logger.warning(f"Deprecation check failed: {e}")
                return {
                    "is_deprecated": True,
                    "deprecation_message": "Deprecated",
                    "replacement": None,
                    "removal_version": None,
                }

        return None

    async def _extract_example_explanation(self, example: str) -> str:
        """Extract explanation from code example."""
        # Look for comments or surrounding text
        lines = example.split("\n")
        comments = []

        for line in lines:
            stripped = line.strip()
            if (
                stripped.startswith("#")
                or stripped.startswith('"""')
                or stripped.startswith("'''")
            ):
                comments.append(stripped)

        if comments:
            return " ".join(comments)

        # Use LLM to generate explanation
        prompt = f"""
        Explain what this code example does:

        {example}

        Provide a brief, clear explanation.
        """

        try:
            response = await self.llm_client.client.chat.completions.create(
                model=self.llm_client.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            self.logger.warning(f"Explanation extraction failed: {e}")
            return "Code example"

    async def _determine_use_case(self, example: str) -> str:
        """Determine the use case for a code example."""
        prompt = f"""
        Determine the use case for this code example:

        {example}

        Return a brief description of what this example demonstrates.
        """

        try:
            response = await self.llm_client.client.chat.completions.create(
                model=self.llm_client.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            self.logger.warning(f"Use case determination failed: {e}")
            return "General usage"

    async def _extract_prerequisites(self, example: str) -> List[str]:
        """Extract prerequisites for a code example."""
        prompt = f"""
        Identify prerequisites needed to run this code example:

        {example}

        Return a JSON array of prerequisite descriptions.
        """

        try:
            response = await self.llm_client.client.chat.completions.create(
                model=self.llm_client.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            result = parse_llm_json_response(content)

            if isinstance(result, dict) and "prerequisites" in result:
                return result["prerequisites"]
            elif isinstance(result, list):
                return result
            else:
                return []

        except Exception as e:
            self.logger.warning(f"Prerequisite extraction failed: {e}")
            return []

    async def _extract_expected_output(self, example: str) -> Optional[str]:
        """Extract expected output from code example."""
        # Look for output in comments
        output_patterns = [
            r"# Output: (.+)",
            r"# Result: (.+)",
            r'"""Output: (.+)"""',
            r"'''Output: (.+)'''",
        ]

        for pattern in output_patterns:
            match = re.search(pattern, example)
            if match:
                return match.group(1)

        return None

    async def _determine_example_complexity(self, example: str) -> str:
        """Determine complexity level of code example."""
        # Use the same complexity calculation as code blocks
        complexity_score = self._calculate_code_complexity(example, "python")

        if complexity_score < 0.3:
            return "beginner"
        elif complexity_score < 0.7:
            return "intermediate"
        else:
            return "advanced"

    def _extract_step_number(self, content: str) -> Optional[int]:
        """Extract step number from tutorial content."""
        # Look for step patterns
        step_patterns = [r"Step\s+(\d+)", r"(\d+)\.\s", r"Step\s+(\d+):"]

        for pattern in step_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    async def _extract_tutorial_prerequisites(self, content: str) -> List[str]:
        """Extract prerequisites from tutorial content."""
        prompt = f"""
        Extract prerequisites from this tutorial step:

        {content}

        Return a JSON array of prerequisite descriptions.
        """

        try:
            response = await self.llm_client.client.chat.completions.create(
                model=self.llm_client.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            result = parse_llm_json_response(content)

            if isinstance(result, dict) and "prerequisites" in result:
                return result["prerequisites"]
            elif isinstance(result, list):
                return result
            else:
                return []

        except Exception as e:
            self.logger.warning(f"Tutorial prerequisite extraction failed: {e}")
            return []

    async def _extract_expected_outcome(self, content: str) -> str:
        """Extract expected outcome from tutorial content."""
        prompt = f"""
        What is the expected outcome of this tutorial step?

        {content}

        Provide a brief description of what should be achieved.
        """

        try:
            response = await self.llm_client.client.chat.completions.create(
                model=self.llm_client.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            self.logger.warning(f"Expected outcome extraction failed: {e}")
            return "Complete the tutorial step"

    async def _determine_tutorial_difficulty(self, content: str) -> str:
        """Determine difficulty level of tutorial content."""
        prompt = f"""
        Determine the difficulty level of this tutorial content:

        {content}

        Classify as: beginner, intermediate, or advanced
        """

        try:
            response = await self.llm_client.client.chat.completions.create(
                model=self.llm_client.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.1,
            )

            difficulty = response.choices[0].message.content.strip().lower()

            if difficulty in ["beginner", "intermediate", "advanced"]:
                return difficulty
            else:
                return "intermediate"

        except Exception as e:
            self.logger.warning(f"Difficulty determination failed: {e}")
            return "intermediate"

    def _extract_time_estimate(self, content: str) -> Optional[str]:
        """Extract time estimate from tutorial content."""
        # Look for time patterns
        time_patterns = [
            r"(\d+)\s*minutes?",
            r"(\d+)\s*hours?",
            r"(\d+)\s*min",
            r"(\d+)\s*hr",
        ]

        for pattern in time_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                time_value = match.group(1)
                if "hour" in pattern or "hr" in pattern:
                    return f"{time_value} hours"
                else:
                    return f"{time_value} minutes"

        return None
