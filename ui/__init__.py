"""Terminal UI layer.

Modules here render output and read input. They must not import the application
core: everything they need is passed in, so the backend stays free of UI
concerns and can be exercised without a terminal.

``common``      shared primitives (ask_choice / confirm / preview_then_confirm)
``chat``        conversation rendering
``settings_ui`` settings menu rendering
``memory_ui``   memory branch rendering
``attachments_ui`` attachment listing
"""
