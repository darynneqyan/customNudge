"""
Native macOS dialog utilities for user input.
Assumption: macOS is the operating system. 
"""

import sys
from typing import Optional
from AppKit import (
    NSApplication, 
    NSAlert, 
    NSInformationalAlertStyle, 
    NSTextField, 
    NSMakeRect, 
    NSAlertFirstButtonReturn
)

def prompt_for_goal() -> Optional[str]:
    """
    Display a native macOS dialog to prompt the user for their goal.
    
    Returns:
        Optional[str]: The goal text entered by the user, or None if cancelled.
    """
    try:
        # For CLI scripts, we need to ensure the app is properly initialized
        app = NSApplication.sharedApplication()
        if app is None:
            print("ERROR: Could not get NSApplication instance", file=sys.stderr)
            return None
        
        app.setActivationPolicy_(0)  
        app.activateIgnoringOtherApps_(True)
        
        alert = NSAlert.alloc().init()
        alert.setMessageText_("Enter Your Goal")
        alert.setInformativeText_("What would you like to achieve today? I will help you stay on track.")
        alert.setAlertStyle_(NSInformationalAlertStyle)
        
        input_field = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 300, 24))
        input_field.setPlaceholderString_("e.g., Focus on deep work, submit the report in 1 hour...")
        input_field.setEditable_(True)
        input_field.setSelectable_(True)
        alert.setAccessoryView_(input_field)
        alert.layout()  
        
        alert.addButtonWithTitle_("Let's do it!")
        alert.addButtonWithTitle_("Cancel")
        
        response = alert.runModal()
        
        from AppKit import NSRunLoop, NSDefaultRunLoopMode, NSDate
        run_loop = NSRunLoop.currentRunLoop()
        for _ in range(3):
            run_loop.runMode_beforeDate_(NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.01))
        
        if response == NSAlertFirstButtonReturn:
            goal_text = input_field.stringValue().strip()
            print(f"DEBUG: First button clicked, goal_text = '{goal_text}'", file=sys.stderr)
            return goal_text if goal_text else None
        else:
            print(f"DEBUG: Second button (Cancel) clicked or other response", file=sys.stderr)
            return None
            
    except Exception as e:
        print(f"ERROR: Failed to show goal dialog: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None

