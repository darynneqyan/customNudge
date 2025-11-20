#!/usr/bin/env python3
"""
GUI for viewing GUM notifications with LLM decisions.
Shows two tabs:
1. Notification Decisions (Yes/No from LLM)
2. Observation Contexts (detailed breakdown)
"""

import tkinter as tk
from tkinter import ttk
import json
from pathlib import Path
from datetime import datetime


class NotificationGUI:
    """GUI for displaying GUM notifications with tabs."""
    
    def __init__(self, user_name: str = "demo_user"):
        """
        Initialize the GUM Notification GUI.
        
        Creates a Tkinter-based GUI for visualizing notification decisions made by the GUM system.
        The GUI monitors JSON files created by the notification system and displays them in real-time.
        
        Args:
            user_name (str): Name of the user being monitored. Used to construct file paths for
                           notification data files. Defaults to "demo_user".
                           
        Attributes:
            user_name (str): The name of the user being monitored
            contexts_file (Path): Path to notification contexts JSON file
            decisions_file (Path): Path to notification decisions JSON file  
            last_count (int): Number of decisions processed in last refresh
            sent_count (int): Number of notifications actually sent
            contexts_displayed (int): Number of contexts displayed in GUI
            root (tk.Tk): Main Tkinter window
            notebook (ttk.Notebook): Tab container for different views
            decisions_frame (tk.Frame): Container for decision widgets
            status_label (tk.Label): Status display widget
            count_label (tk.Label): Counter display widget
        """
        self.user_name = user_name
        self.contexts_file = Path(f"notification_contexts_{user_name.lower().replace(' ', '_')}.json")
        self.decisions_file = Path(f"notification_decisions_{user_name.lower().replace(' ', '_')}.json")
        self.last_count = 0
        self.sent_count = 0
        self.contexts_displayed = 0
        
        # Create main window
        self.root = tk.Tk()
        self.root.title(f"🔔 GUM Notifications - {user_name}")
        self.root.geometry("950x850")
        
        self._setup_ui()
    
    def _setup_ui(self):
        """
        Set up the complete UI layout for the notification GUI.
        
        Creates the main interface with:
        - Header with title and branding
        - Status display showing current system state
        - Tabbed interface for different data views
        - Control buttons for manual refresh
        - Counter display for statistics
        """
        # Header
        header = tk.Frame(self.root, bg="#2c3e50", height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        title = tk.Label(
            header,
            text="🔔 customNudge Decisions",
            font=("Arial", 20, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title.pack(pady=15)
        
        # Status
        status_frame = tk.Frame(self.root, bg="#ecf0f1")
        status_frame.pack(fill=tk.X)
        
        self.status_label = tk.Label(
            status_frame,
            text="Watching for notifications...",
            font=("Arial", 10),
            bg="#ecf0f1",
            fg="#7f8c8d"
        )
        self.status_label.pack(pady=5)
        
        # Tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Decisions
        self.setup_decisions_tab()
        
        # Tab 2: Contexts
        # self.setup_contexts_tab()
        
        # Buttons
        button_frame = tk.Frame(self.root, bg="#ecf0f1")
        button_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(
            button_frame,
            text="🔄 Refresh",
            command=self.refresh,
            bg="#3498db",
            fg="black",
            font=("Arial", 10, "bold"),
            padx=15
        ).pack(side=tk.LEFT, padx=10)
        
        self.count_label = tk.Label(
            button_frame,
            text="Decisions: 0 | Sent: 0",
            font=("Arial", 10, "bold"),
            bg="#ecf0f1"
        )
        self.count_label.pack(side=tk.RIGHT, padx=10)
    
    def setup_decisions_tab(self):
        """Setup the Notification Decisions tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📋")
        
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        self.decisions_frame = tk.Frame(canvas)
        
        self.decisions_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.decisions_frame, anchor="nw", width=900)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Store canvas reference for scroll position management
        self.decisions_canvas = canvas
    
    def setup_contexts_tab(self):
        """Setup the Observation Contexts tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🔍 Observation Contexts")
        
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        self.contexts_frame = tk.Frame(canvas)
        
        self.contexts_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.contexts_frame, anchor="nw", width=900)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def add_decision(self, decision_data: dict):
        """
        Add a notification decision widget to the decisions tab.
        
        Creates a comprehensive display widget for each notification decision made by the GUM system.
        The widget shows decision metadata, reasoning, and context in a visually organized format.
        
        Args:
            decision_data (dict): Dictionary containing decision information with keys:
                - should_notify (bool): Whether notification was sent
                - timestamp (str): ISO timestamp of the decision
                - notification_type (str): Type of notification (focus, break, habit, etc.)
                - relevance_score (float): Relevance score 1-10
                - urgency_score (float): Urgency score 1-10  
                - impact_score (float): Impact score 1-10
                - reasoning (str): LLM reasoning for the decision
                - notification_message (str): Message sent (if any)
                - observation_content (str): Original observation text
                - generated_propositions_count (int): Number of propositions generated
                - similar_propositions_count (int): Number of similar propositions found
                - similar_observations_count (int): Number of similar observations found
                
        Visual Design:
        - Green header for sent notifications, gray for not sent
        - Color-coded score display with emoji indicators
        - Scrollable text areas for long content
        - Responsive layout with proper text wrapping
        """
        should_notify = decision_data.get('should_notify', False)
        
        # Determine color based on decision
        if should_notify:
            header_bg = "#27ae60"  # Green for YES
            header_text_prefix = "✅ SENT"
        else:
            header_bg = "#95a5a6"  # Gray for NO
            header_text_prefix = "❌ NOT SENT"
        
        frame = tk.Frame(self.decisions_frame, bg="white", relief="solid", borderwidth=2, padx=15, pady=10)
        frame.pack(fill=tk.X, pady=5)
        
        # Timestamp and decision
        timestamp = decision_data.get('timestamp', '')
        try:
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%I:%M:%S %p")
        except:
            time_str = timestamp
        
        notif_type = decision_data.get('notification_type', 'none').upper()
        header_text = f"{header_text_prefix} - {notif_type} - {time_str}"
        
        tk.Label(
            frame,
            text=header_text,
            font=("Arial", 11, "bold"),
            bg=header_bg,
            fg="white",
            padx=10,
            pady=5
        ).pack(fill=tk.X)
        
        # Scores
        rel = decision_data.get('relevance_score', 0)
        urg = decision_data.get('urgency_score', 0)
        imp = decision_data.get('impact_score', 0)
        
        tk.Label(
            frame,
            text=f"📊 Relevance: {rel}/10 | Urgency: {urg}/10 | Impact: {imp}/10",
            font=("Arial", 9, "bold"),
            bg="#ecf0f1",
            padx=5,
            pady=3
        ).pack(fill=tk.X, pady=(5, 0))
        
        # Effectiveness evaluation (if available)
        effectiveness_score = decision_data.get('effectiveness_score')
        if effectiveness_score is not None:
            # Determine color based on effectiveness
            if effectiveness_score == 1.0:
                eff_color = "#27ae60"  # Green for effective
                eff_emoji = "✅"
            elif effectiveness_score == 0.5:
                eff_color = "#f39c12"  # Orange for partially effective
                eff_emoji = "⚠️"
            else:
                eff_color = "#e74c3c"  # Red for ineffective
                eff_emoji = "❌"
            
            compliance_percentage = decision_data.get('compliance_percentage', 0)
            compliance_pattern = decision_data.get('compliance_pattern', 'Unknown')
            evaluation_source = decision_data.get('evaluation_source', 'unknown')
            
            eff_frame = tk.Frame(frame, bg=eff_color, padx=10, pady=5)
            eff_frame.pack(fill=tk.X, pady=(5, 0))
            
            eff_text = f"{eff_emoji} Effectiveness: {effectiveness_score}/1.0"
            # Always show compliance information if available
            if compliance_percentage is not None:
                eff_text += f" | Compliance: {compliance_percentage:.1f}% ({compliance_pattern})"
            if evaluation_source:
                source_text = "System Capture" if evaluation_source == 'system_capture' else "Observations"
                eff_text += f" | Source: {source_text}"
            
            tk.Label(
                eff_frame,
                text=eff_text,
                font=("Arial", 10, "bold"),
                bg=eff_color,
                fg="white",
                wraplength=850,
                justify="left"
            ).pack(anchor="w")
            
            # Effectiveness reasoning
            eff_reasoning = decision_data.get('effectiveness_reasoning', '')
            if eff_reasoning:
                tk.Label(
                    frame,
                    text=f"💡 Effectiveness Reasoning: {eff_reasoning}",
                    font=("Arial", 9),
                    bg="white",
                    wraplength=850,
                    justify="left",
                    fg="#34495e"
                ).pack(anchor="w", pady=(5, 0))
        
        # Notification message (if sent)
        if should_notify:
            message = decision_data.get('notification_message', '')
            msg_frame = tk.Frame(frame, bg="#d5f4e6", padx=10, pady=8)
            msg_frame.pack(fill=tk.X, pady=(5, 0))
            
            tk.Label(
                msg_frame,
                text=f"💬 {message}",
                font=("Arial", 12, "bold"),
                bg="#d5f4e6",
                fg="#27ae60",
                wraplength=850,
                justify="left"
            ).pack(anchor="w")
        
        # Reasoning
        reasoning = decision_data.get('reasoning', '')
        tk.Label(
            frame,
            text=f"💭 Reasoning: {reasoning}",
            font=("Arial", 14),
            bg="white",
            wraplength=850,
            justify="left",
            fg="#2c3e50"
        ).pack(anchor="w", pady=(5, 0))
        
        # Context summary
        obs_preview = decision_data.get('observation_content', '')[:100]
        gen_count = decision_data.get('generated_propositions_count', 0)
        sim_props_count = decision_data.get('similar_propositions_count', 0)
        sim_obs_count = decision_data.get('similar_observations_count', 0)
        
        tk.Label(
            frame,
            text=f"📝 Observation: {obs_preview}...",
            font=("Arial", 13),
            bg="white",
            wraplength=850,
            justify="left",
            fg="#34495e"
        ).pack(anchor="w", pady=(5, 0))
        
        tk.Label(
            frame,
            text=f"📚 Context: {gen_count} generated | {sim_props_count} similar propositions | {sim_obs_count} similar observations",
            font=("Arial", 12),
            bg="white",
            fg="#7f8c8d"
        ).pack(anchor="w", pady=(2, 0))
    
    def add_context(self, context: dict):
        """Add full observation context to Tab 2."""
        frame = tk.Frame(self.contexts_frame, bg="white", relief="solid", borderwidth=1, padx=15, pady=10)
        frame.pack(fill=tk.X, pady=5)
        
        # Timestamp and ID
        timestamp = context.get('timestamp', '')
        obs_id = context.get('observation_id', 'N/A')
        
        try:
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%I:%M:%S %p on %b %d")
        except:
            time_str = timestamp
        
        tk.Label(
            frame,
            text=f"Observation #{obs_id} - {time_str}",
            font=("Arial", 11, "bold"),
            bg="white",
            fg="#2c3e50"
        ).pack(anchor="w")
        
        # Observation content
        tk.Label(
            frame,
            text="📝 Observation:",
            font=("Arial", 9, "bold"),
            bg="white"
        ).pack(anchor="w", pady=(10, 2))
        
        obs_text = tk.Text(frame, height=4, width=100, wrap="word", font=("Arial", 14), bg="#f8f9fa")
        obs_text.insert("1.0", context.get('observation_content', '')[:400] + "...")
        obs_text.config(state="disabled")
        obs_text.pack(fill=tk.X, pady=2)
        
        # Generated Propositions
        gen_props = context.get('generated_propositions', [])
        if gen_props:
            tk.Label(
                frame,
                text=f"💡 Generated Propositions ({len(gen_props)}):",
                font=("Arial", 9, "bold"),
                bg="white"
            ).pack(anchor="w", pady=(10, 2))
            
            for i, prop in enumerate(gen_props[:3], 1):
                prop_frame = tk.Frame(frame, bg="#e8f5e9", padx=5, pady=3)
                prop_frame.pack(fill=tk.X, pady=2)
                
                tk.Label(
                    prop_frame,
                    text=f"{i}. [{prop.get('confidence', 0)}/10] {prop.get('text', '')}",
                    font=("Arial", 13),
                    bg="#e8f5e9",
                    wraplength=850,
                    justify="left"
                ).pack(anchor="w")
        
        # Similar Propositions
        sim_props = context.get('similar_propositions', [])
        if sim_props:
            tk.Label(
                frame,
                text=f"🔗 Similar Propositions ({len(sim_props)}):",
                font=("Arial", 9, "bold"),
                bg="white"
            ).pack(anchor="w", pady=(10, 2))
            
            for i, prop in enumerate(sim_props[:3], 1):
                prop_frame = tk.Frame(frame, bg="#fff3cd", padx=5, pady=3)
                prop_frame.pack(fill=tk.X, pady=2)
                
                tk.Label(
                    prop_frame,
                    text=f"{i}. [{prop.get('confidence', 0)}/10, Rel: {prop.get('relevance_score', 0):.2f}] {prop.get('text', '')[:150]}...",
                    font=("Arial", 13),
                    bg="#fff3cd",
                    wraplength=850,
                    justify="left"
                ).pack(anchor="w")
        
        # Similar Observations
        sim_obs = context.get('similar_observations', [])
        if sim_obs:
            tk.Label(
                frame,
                text=f"📊 Similar Observations ({len(sim_obs)}):",
                font=("Arial", 9, "bold"),
                bg="white"
            ).pack(anchor="w", pady=(10, 2))
            
            for i, obs in enumerate(sim_obs[:3], 1):
                obs_frame = tk.Frame(frame, bg="#e3f2fd", padx=5, pady=3)
                obs_frame.pack(fill=tk.X, pady=2)
                
                tk.Label(
                    obs_frame,
                    text=f"{i}. [Rel: {obs.get('relevance_score', 0):.2f}] {obs.get('content', '')[:150]}...",
                    font=("Arial", 13),
                    bg="#e3f2fd",
                    wraplength=850,
                    justify="left"
                ).pack(anchor="w")
    
    def clear_all_widgets(self):
        """Clear all widgets from both tabs."""
        for widget in self.decisions_frame.winfo_children():
            widget.destroy()
        # for widget in self.contexts_frame.winfo_children():
        #     widget.destroy()
        self.contexts_displayed = 0
        self.sent_count = 0

    def refresh(self):
        """
        Refresh the GUI by loading new notification data from JSON files.
        
        This is the core data loading function that:
        1. Reads the decisions JSON file created by the notification system
        2. Compares current count with last known count to detect new decisions
        3. Clears existing widgets and reloads all data in reverse chronological order
        4. Updates counters and status displays
        5. Handles file I/O errors gracefully
        
        The refresh process is designed to be idempotent - running it multiple times
        with the same data will produce the same result. New data is detected by
        comparing the length of the decisions array.
        
        Data Flow:
        - Reads from notification_decisions_{user_name}.json
        - Parses JSON array of decision objects
        - Creates GUI widgets for each decision
        - Updates statistics counters
        - Shows error messages for file I/O issues
        
        Error Handling:
        - Catches JSON parsing errors
        - Handles missing files gracefully
        - Updates status display with error messages
        - Continues operation even if individual files fail
        """
        # Capture refresh time at the start of refresh
        refresh_time = datetime.now()
        
        # Check for new decisions
        if self.decisions_file.exists():
            try:
                with open(self.decisions_file, 'r') as f:
                    decisions = json.load(f)
                
                current_count = len(decisions)
                previous_count = self.last_count
                
                # Save current scroll position before clearing
                if hasattr(self, 'decisions_canvas'):
                    scroll_position = self.decisions_canvas.yview()[0]  # Get current scroll position (0.0 to 1.0)
                else:
                    scroll_position = 0.0
                
                # Always refresh when refresh button is clicked or if there are new decisions
                self.clear_all_widgets()
                
                # Load all decisions in reverse order (newest first)
                for decision in reversed(decisions):
                    self.add_decision(decision)
                    if decision.get('should_notify'):
                        self.sent_count += 1
                
                # Restore scroll position after widgets are added
                if hasattr(self, 'decisions_canvas'):
                    self.decisions_frame.update_idletasks()
                    # Restore scroll position
                    self.decisions_canvas.yview_moveto(scroll_position)
                
                new_decisions = current_count - previous_count
                self.last_count = current_count
                self.count_label.config(text=f"Decisions: {current_count} | Sent: {self.sent_count}")
                
                # Status messaging for 3 cases - use refresh_time instead of datetime.now()
                refresh_time_str = refresh_time.strftime('%I:%M:%S %p')
                if new_decisions > 0:
                    self.status_label.config(
                        text=f"✅ {new_decisions} new decision(s) | Last updated: {refresh_time_str}",
                        fg="#27ae60"
                    )
                elif new_decisions == 0 and current_count > 0:
                    self.status_label.config(
                        text=f"✅ Refreshed | {current_count} total decision(s) | Last updated: {refresh_time_str}",
                        fg="#27ae60"
                    )
                else:
                    self.status_label.config(
                        text=f"✅ No decisions yet | Last updated: {refresh_time_str}",
                        fg="#7f8c8d"
                    )
            except Exception as e:
                self.status_label.config(text=f"Error loading decisions: {e}", fg="#e74c3c")
        
        # # Check for new contexts
        # if self.contexts_file.exists():
        #     try:
        #         with open(self.contexts_file, 'r') as f:
        #             contexts = json.load(f)
                
        #         # Load all contexts in reverse order (newest first)
        #         for context in reversed(contexts):
        #             self.add_context(context)
        #             self.contexts_displayed += 1
                    
        #     except Exception as e:
        #         self.status_label.config(text=f"Error loading contexts: {e}", fg="#e74c3c")
        
        if not self.decisions_file.exists() and not self.contexts_file.exists():
            self.status_label.config(text="Waiting for notification files...", fg="#e67e22")
    
    def auto_refresh(self):
        """
        Set up automatic refresh of the GUI every 3 seconds.
        
        This method creates a recursive timer that:
        1. Calls the refresh() method to load new data
        2. Schedules itself to run again in 3 seconds
        3. Continues indefinitely until the GUI is closed
        
        The auto-refresh ensures the GUI stays synchronized with the notification
        system without requiring manual intervention. The 3-second interval provides
        a good balance between responsiveness and system resource usage.
        
        Implementation:
        - Uses Tkinter's after() method for non-blocking timer
        - Recursive call pattern for continuous operation
        - No cleanup needed - timer stops when window closes
        """
        self.refresh()
        self.root.after(3000, self.auto_refresh)
    
    def run(self):
        """
        Start the GUI application and begin the main event loop.
        
        This method initializes the GUI and starts the Tkinter main loop:
        1. Performs initial data refresh to load existing notifications
        2. Starts the auto-refresh timer for continuous updates
        3. Enters the Tkinter main loop (blocking call)
        
        The main loop handles all user interactions, window events, and timer
        callbacks until the application is closed. This is the entry point for
        the GUI application.
        
        Lifecycle:
        - Initial refresh loads any existing notification data
        - Auto-refresh timer starts for continuous updates
        - Main loop runs until window is closed
        - All cleanup is handled automatically by Tkinter
        """
        self.refresh()
        self.auto_refresh()
        self.root.mainloop()


def main():
    """
    Main entry point for the GUM Notification GUI application.
    
    This function handles command-line arguments, user input, and application startup:
    1. Parses command-line arguments for user name
    2. Prompts for user name if not provided
    3. Displays startup information and instructions
    4. Creates and runs the GUI application
    
    Command Line Usage:
        python notification_gui.py [user_name]
        
    If no user_name is provided, the application will prompt for input.
    The user_name is used to construct file paths for notification data files.
    
    Prerequisites:
    - GUM system must be running with --enable-notifications flag
    - Notification system must be creating JSON files
    - Proper API keys must be configured for the GUM system
    
    Example:
        python notification_gui.py "John Doe"
        # Watches for files: notification_decisions_john_doe.json
    """
    import sys
    
    print("GUM Notification GUI")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        user_name = sys.argv[1]
    else:
        user_name = input("Enter user name: ").strip() or "demo_user"
    
    print(f"Starting GUI for: {user_name}")
    print(f"Watching: notification_contexts_{user_name.lower().replace(' ', '_')}.json")
    print("=" * 60)
    print("\n✨ GUI window opening...")
    print("💡 Make sure GUM is running with --enable-notifications")
    print(f"   python -m gum.cli -u '{user_name}' --enable-notifications\n")
    
    gui = NotificationGUI(user_name)
    gui.run()


if __name__ == "__main__":
    main()