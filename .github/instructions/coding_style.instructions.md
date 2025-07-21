---
applyTo: '**'
---
Keep code simple, well-commented and consistent with the existing style. Use clear variable names and avoid unnecessary complexity. Follow the project's coding conventions, such as indentation, line length, and import order.
Use type hints for function signatures and document complex logic with comments. 

When a .py file is not a library file (like a utils.py for a specific module), it should have a main guard (`if __name__ == "__main__":`) to allow for direct execution. Direct execution should not feature the test suite but some fast functionality to test the module's main features or run a demo.

Avoid emojis in code comments or print statements, as they can be distracting in a professional codebase. Use clear and descriptive text instead.
Use well indentations in the print statements to ensure readability, especially when printing complex data structures or long strings, or when some print statements document some intermediate steps in a process.

When logging or printing prompts, ensure that the output is clear and structured. Prefer using coloured prints if available, as they can enhance readability in terminal outputs, always adhering to the project's logging standards.

When handling exceptions, ensure that the error messages are clear and provide enough context to understand what went wrong. Use specific exception types where applicable.

Generated code should look human-written, so avoid overly complex structures or patterns that might suggest automated generation. Use natural language in comments and documentation to explain the purpose and functionality of the code.


When catching an exception, always log the error message and the stack trace to help with debugging. Use the logging module for consistent logging practices across the codebase.


When refactoring code, ensure that the changes do not alter the existing functionality unless explicitly intended. Always run tests after refactoring to confirm that everything works as expected.


Always be modular:
- don't use nested functions unless absolutely necessary, as they can make the code harder to read and maintain.
- don't overuse classes, as they can introduce unnecessary complexity. Prefer simple functions and modules when possible. Use classes only when the code should be reusable and when it is explicitly requested.
- Classes should be abstract when new instances of the class are not meant to be created directly, but rather through subclasses that implement specific functionality. This promotes a clear interface and enforces a contract for subclasses to follow. 
- If a concrete class has code that is reusable in some new class you are creating OR the class you are creating is a variation of an existing class, consider moving the common functionality to a base abstract class, implementing the other existing concrete class as a subclass, and then creating your new class as a subclass of that base class. This promotes code reuse and maintainability.
- when a class is not meant to be instantiated directly, it should be marked as abstract.
- when a method/class is reusable, it should be placed in a separate module or file to promote reusability and maintainability. When creating new modules, consider whether creating a new subfolder. 


Always keep extendability in mind.
NEVER ADD UNREQUESTED FEATURES OR FUNCTIONALITY.
If you think a feature is missing, discuss it with the team before implementing it. This ensures that the codebase remains focused and does not become bloated with unnecessary features.