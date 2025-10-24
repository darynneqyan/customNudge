#!/usr/bin/env python3
"""
System State Capture Module for Adaptive Nudge Engine

Captures structured text-based representation of user's computing environment
for post-nudge effectiveness evaluation.

Goal: Replace visual screenshots with structured data that can be analyzed
by LLM to determine if user acted on a nudge.
"""

import subprocess
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

class SystemStateCapture:
    """
    Captures comprehensive system state for post-nudge evaluation.
    
    This class provides methods to capture various aspects of the user's
    computing environment including active applications, browser tabs,
    clipboard history, and open files.
    """
    
    def __init__(self):
        """
        Initialize the system state capture module.
        
        Sets up logging and prepares for system state capture operations.
        """
        self.logger = logging.getLogger("gum.adaptive_nudge.state_capture")
        self.logger.info("SystemStateCapture initialized")
    
    def capture_system_state(self) -> Dict[str, Any]:
        """
        Capture comprehensive system state snapshot.
        
        This is the main method that orchestrates the capture of all
        relevant system state information for post-nudge evaluation.
        
        Returns:
            Dict containing:
            - timestamp: ISO timestamp of capture
            - active_app: Current foreground application name
            - window_title: Active window title
            - browser_tabs: List of open browser tabs with URLs
            - clipboard_history: Recent clipboard entries
            - open_files: Currently open files in editors
            - system_info: Basic system information
            - recent_apps: Recently used applications
            - error: Any errors encountered during capture
        """
        try:
            self.logger.info("Starting system state capture")
            
            state = {
                "timestamp": datetime.now().isoformat(),
                "active_app": self._get_active_application(),
                "window_title": self._get_window_title(),
                "browser_tabs": self._get_browser_tabs(),
                "clipboard_history": self._get_clipboard_history(),
                "open_files": self._get_open_files(),
                "system_info": self._get_system_info(),
                "recent_apps": self._get_recent_applications()
            }
            
            self.logger.info("System state captured successfully")
            return state
            
        except Exception as e:
            self.logger.error(f"Error capturing system state: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "active_app": None,
                "window_title": None,
                "browser_tabs": [],
                "clipboard_history": [],
                "open_files": [],
                "system_info": {},
                "recent_apps": []
            }
    
    def _get_active_application(self) -> Optional[str]:
        """
        Get the currently active application name.
        
        Uses AppleScript to query System Events for the frontmost application.
        
        Returns:
            str: Name of the active application, or None if error
        """
        try:
            result = subprocess.run([
                "osascript", "-e", 
                "tell application \"System Events\" to get name of first application process whose frontmost is true"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                app_name = result.stdout.strip()
                self.logger.debug(f"Active application: {app_name}")
                return app_name
            else:
                self.logger.warning(f"Could not get active application: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.warning("Timeout getting active application")
            return None
        except Exception as e:
            self.logger.warning(f"Could not get active application: {e}")
            return None
    
    def _get_window_title(self) -> Optional[str]:
        """
        Get the title of the currently active window.
        
        Uses AppleScript to query the frontmost window title.
        
        Returns:
            str: Window title, or None if error
        """
        try:
            result = subprocess.run([
                "osascript", "-e",
                "tell application \"System Events\" to get name of first window of first application process whose frontmost is true"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                title = result.stdout.strip()
                self.logger.debug(f"Window title: {title}")
                return title
            else:
                self.logger.warning(f"Could not get window title: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.warning("Timeout getting window title")
            return None
        except Exception as e:
            self.logger.warning(f"Could not get window title: {e}")
            return None
    
    def _get_browser_tabs(self) -> List[Dict[str, str]]:
        """
        Get open browser tabs from Safari, Chrome, and Firefox.
        
        Attempts to capture tabs from multiple browsers to get comprehensive
        view of user's web activity.
        
        Returns:
            List of dicts with browser, url, and index for each tab
        """
        tabs = []
        
        # Safari tabs
        safari_tabs = self._get_safari_tabs()
        tabs.extend(safari_tabs)
        
        # Chrome tabs
        chrome_tabs = self._get_chrome_tabs()
        tabs.extend(chrome_tabs)
        
        # Firefox tabs
        firefox_tabs = self._get_firefox_tabs()
        tabs.extend(firefox_tabs)
        
        self.logger.debug(f"Found {len(tabs)} browser tabs")
        return tabs
    
    def _get_safari_tabs(self) -> List[Dict[str, str]]:
        """Get Safari browser tabs."""
        tabs = []
        try:
            result = subprocess.run([
                "osascript", "-e",
                "tell application \"Safari\" to get URL of every tab of every window"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                urls = result.stdout.strip().split(", ")
                for i, url in enumerate(urls):
                    if url and url != "missing value" and url.strip():
                        tabs.append({
                            "browser": "Safari", 
                            "url": url.strip(), 
                            "index": i
                        })
        except Exception as e:
            self.logger.warning(f"Could not get Safari tabs: {e}")
        
        return tabs
    
    def _get_chrome_tabs(self) -> List[Dict[str, str]]:
        """Get Chrome browser tabs."""
        tabs = []
        try:
            result = subprocess.run([
                "osascript", "-e",
                "tell application \"Google Chrome\" to get URL of every tab of every window"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                urls = result.stdout.strip().split(", ")
                for i, url in enumerate(urls):
                    if url and url != "missing value" and url.strip():
                        tabs.append({
                            "browser": "Chrome", 
                            "url": url.strip(), 
                            "index": i
                        })
        except Exception as e:
            self.logger.warning(f"Could not get Chrome tabs: {e}")
        
        return tabs
    
    def _get_firefox_tabs(self) -> List[Dict[str, str]]:
        """Get Firefox browser tabs."""
        tabs = []
        try:
            result = subprocess.run([
                "osascript", "-e",
                "tell application \"Firefox\" to get URL of every tab of every window"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                urls = result.stdout.strip().split(", ")
                for i, url in enumerate(urls):
                    if url and url != "missing value" and url.strip():
                        tabs.append({
                            "browser": "Firefox", 
                            "url": url.strip(), 
                            "index": i
                        })
        except Exception as e:
            self.logger.warning(f"Could not get Firefox tabs: {e}")
        
        return tabs
    
    def _get_clipboard_history(self) -> List[str]:
        """
        Get recent clipboard history.
        
        Currently captures only the current clipboard content.
        For full history, would need integration with clipboard managers.
        
        Returns:
            List of clipboard entries
        """
        try:
            result = subprocess.run([
                "osascript", "-e", "the clipboard"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and result.stdout.strip():
                content = result.stdout.strip()
                self.logger.debug(f"Clipboard content: {content[:50]}...")
                return [content]
            return []
            
        except subprocess.TimeoutExpired:
            self.logger.warning("Timeout getting clipboard")
            return []
        except Exception as e:
            self.logger.warning(f"Could not get clipboard: {e}")
            return []
    
    def _get_open_files(self) -> List[Dict[str, str]]:
        """
        Get currently open files in code editors.
        
        Checks VS Code, Xcode, and other common editors for open files.
        
        Returns:
            List of dicts with editor and file path
        """
        open_files = []
        
        # VS Code
        vscode_files = self._get_vscode_files()
        open_files.extend(vscode_files)
        
        # Xcode
        xcode_files = self._get_xcode_files()
        open_files.extend(xcode_files)
        
        # Sublime Text
        sublime_files = self._get_sublime_files()
        open_files.extend(sublime_files)
        
        self.logger.debug(f"Found {len(open_files)} open files")
        return open_files
    
    def _get_vscode_files(self) -> List[Dict[str, str]]:
        """Get VS Code open files."""
        files = []
        try:
            result = subprocess.run([
                "osascript", "-e",
                "tell application \"Visual Studio Code\" to get path of every document"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                paths = result.stdout.strip().split(", ")
                for path in paths:
                    if path and path != "missing value" and path.strip():
                        files.append({
                            "editor": "VS Code", 
                            "path": path.strip()
                        })
        except Exception as e:
            self.logger.warning(f"Could not get VS Code files: {e}")
        
        return files
    
    def _get_xcode_files(self) -> List[Dict[str, str]]:
        """Get Xcode open files."""
        files = []
        try:
            result = subprocess.run([
                "osascript", "-e",
                "tell application \"Xcode\" to get path of every document"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                paths = result.stdout.strip().split(", ")
                for path in paths:
                    if path and path != "missing value" and path.strip():
                        files.append({
                            "editor": "Xcode", 
                            "path": path.strip()
                        })
        except Exception as e:
            self.logger.warning(f"Could not get Xcode files: {e}")
        
        return files
    
    def _get_sublime_files(self) -> List[Dict[str, str]]:
        """Get Sublime Text open files."""
        files = []
        try:
            result = subprocess.run([
                "osascript", "-e",
                "tell application \"Sublime Text\" to get path of every document"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                paths = result.stdout.strip().split(", ")
                for path in paths:
                    if path and path != "missing value" and path.strip():
                        files.append({
                            "editor": "Sublime Text", 
                            "path": path.strip()
                        })
        except Exception as e:
            self.logger.warning(f"Could not get Sublime Text files: {e}")
        
        return files
    
    def _get_system_info(self) -> Dict[str, str]:
        """
        Get basic system information.
        
        Returns:
            Dict with user, hostname, and uptime information
        """
        try:
            system_info = {
                "user": os.getenv("USER", "unknown"),
                "hostname": self._get_hostname(),
                "uptime": self._get_uptime()
            }
            return system_info
        except Exception as e:
            self.logger.warning(f"Could not get system info: {e}")
            return {}
    
    def _get_hostname(self) -> str:
        """Get system hostname."""
        try:
            result = subprocess.run(["hostname"], capture_output=True, text=True, timeout=5)
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"
    
    def _get_uptime(self) -> str:
        """Get system uptime."""
        try:
            result = subprocess.run(["uptime"], capture_output=True, text=True, timeout=5)
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"
    
    def _get_recent_applications(self) -> List[str]:
        """
        Get recently used applications.
        
        Returns:
            List of application names currently running
        """
        try:
            result = subprocess.run([
                "osascript", "-e",
                "tell application \"System Events\" to get name of every application process"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                apps = result.stdout.strip().split(", ")
                app_list = [app.strip() for app in apps if app.strip()]
                self.logger.debug(f"Found {len(app_list)} running applications")
                return app_list
            return []
            
        except subprocess.TimeoutExpired:
            self.logger.warning("Timeout getting recent applications")
            return []
        except Exception as e:
            self.logger.warning(f"Could not get recent applications: {e}")
            return []
