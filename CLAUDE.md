# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

hCaptcha Challenger is a Python library that solves hCaptcha challenges using multimodal large language models and computer vision techniques. It uses browser automation via Playwright and integrates with Google Gemini API for visual reasoning.

## Development Commands

## Architecture Overview

### Agent-Based System
The core architecture revolves around agents (`src/hcaptcha_challenger/agent/`):
- **AgentV**: Main challenge-solving agent that orchestrates browser automation and AI reasoning
- **Collector**: Automated dataset collection agent
- **AgentConfig**: Configuration management for agent behavior and model selection

### Tool Plugin System
Tools in `src/hcaptcha_challenger/tools/` are pluggable components for different challenge types:
- **ChallengeClassifier**: Determines challenge type from screenshots
- **ImageClassifier**: Handles grid-based image selection challenges
- **Spatial*Reasoner**: Family of tools for spatial reasoning (point, bbox, path)

### Challenge Processing Flow
1. Browser automation captures challenge screenshot
2. ChallengeClassifier identifies challenge type
3. Appropriate tool is selected based on challenge type
4. Tool processes image using Gemini API or specialized models
5. Response is formatted and submitted to browser

### Configuration Pattern
- Uses Pydantic models extensively for validation (`models.py`)
- Environment variables managed via pydantic-settings
- Prompt templates centralized in `prompts.py`

## Key Implementation Details

### Async Programming
The codebase heavily uses async/await patterns. When modifying:
- Maintain async context throughout the call chain
- Use `asyncio.create_task()` for concurrent operations
- Handle browser lifecycle properly with async context managers

### Gemini API Integration
- API key management through environment variables
- Cost tracking implemented in `helper/cost_calculator.py`
- Retry logic and error handling in tool classes
- Temperature and other model parameters configurable

### Browser Automation
- Playwright-based with support for Chromium
- Mouse movement visualization for debugging
- Session recording capabilities
- Stealth techniques to avoid detection

### Spatial Reasoning
- Coordinate grid system for visual reasoning
- Support for point, bounding box, and path operations
- Visual debugging through attention point visualization

## Testing Approach

### Test Data Organization
- Challenge images organized by type in `tests/challenge_view/`
- Coordinate grids for spatial reasoning validation
- Real challenge examples for integration testing

### Test Patterns
- Async test fixtures for browser setup
- Mock Gemini API responses where appropriate
- Visual regression testing with saved images
- Schema validation tests for data models

## Common Patterns to Follow

### Adding New Challenge Types
1. Define data models in `models.py`
2. Create tool class in `tools/` directory
3. Update ChallengeClassifier to recognize new type
4. Add prompt templates to `prompts.py`
5. Write tests with example images

### Modifying Agent Behavior
1. Update AgentConfig in `models.py` for new parameters
2. Modify AgentV class methods in `agent/challenger.py`
3. Ensure browser lifecycle management is maintained
4. Add logging for debugging

### Working with Prompts
1. Keep prompts in `prompts.py` for centralization
2. Use structured output formats (JSON)
3. Include visual reasoning chains for transparency
4. Test prompts with diverse challenge examples