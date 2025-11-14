# IGNORE THIS SHIT

# PyObjC imports for macOS notifications with buttons
try:
    from Foundation import NSUserNotification, NSUserNotificationCenter, NSObject
    from AppKit import NSApplication
    import objc
    
    class NotificationDelegate(NSObject):
        """Delegate for handling notification button clicks."""
        
        def userNotificationCenter_didActivateNotification_(self, center, notification):
            """Handle notification interaction."""
            try:
                user_info = notification.userInfo()
                nudge_id = user_info.get("nudge_id", "unknown") if user_info else "unknown"
                activation_type = notification.activationType()
                
                # Get the notifier instance from the delegate
                if hasattr(self, 'notifier') and self.notifier:
                    if activation_type == 1:  # Action button (Thanks!)
                        feedback = "thanks"
                    elif activation_type == 2:  # Other button (Not now!)
                        feedback = "not_now"
                    else:
                        feedback = "no_response"
                    
                    # Store feedback
                    if not hasattr(self.notifier, 'button_feedback'):
                        self.notifier.button_feedback = {}
                    if not hasattr(self.notifier, 'session_stats'):
                        self.notifier.session_stats = {
                            "thanks_count": 0,
                            "not_now_count": 0,
                            "no_response_count": 0,
                            "total_notifications": 0
                        }
                    
                    self.notifier.button_feedback[nudge_id] = feedback
                    self.notifier.session_stats[f"{feedback}_count"] = self.notifier.session_stats.get(f"{feedback}_count", 0) + 1
                    self.notifier.logger.info(f"📊 User feedback for {nudge_id}: {feedback}")
                    self.notifier._update_satisfaction_multiplier()
                    
            except Exception as e:
                if hasattr(self, 'notifier') and self.notifier:
                    self.notifier.logger.error(f"Error in notification delegate: {e}")
    
    PYOBJC_AVAILABLE = True
    
except ImportError:
    PYOBJC_AVAILABLE = False
    NotificationDelegate = None

    # tHIS  SHIT IS FOR MODAL

    # this is the notif display with buttons, but modal pop up window
    # def _display_notification_with_buttons(self, message: str, notification_type: str, nudge_id: str):
    #     """Display a native macOS notification with feedback buttons."""
    #     try:
    #         escaped_message = message.replace('"', '\\"').replace("'", "\\'")
            
    #         title_map = {
    #             'break': '⏰ Break Reminder',
    #             'focus': '🎯 Focus Nudge', 
    #             'productivity': '⚡ Productivity Tip',
    #             'habit': '🔄 Habit Reminder',
    #             'health': '💚 Health Nudge',
    #             'general': '🔔 GUM Nudge'
    #         }
            
    #         title = title_map.get(notification_type, '🔔 GUM Nudge')
    #         escaped_title = title.replace('"', '\\"').replace("'", "\\'")
            
    #         # AppleScript with buttons
    #         script = f'''
    #         try
    #             set theResponse to button returned of (display alert "{escaped_title}" message "{escaped_message}" buttons {{"Thanks!", "Not now!"}} default button "Thanks!" giving up after 30)
    #             return theResponse
    #         on error
    #             return "no_response"
    #         end try
    #         '''
            
    #         # Execute and capture button click
    #         def show_alert_async():
    #             try:
    #                 result = subprocess.run(
    #                     ['osascript', '-e', script], 
    #                     capture_output=True, 
    #                     text=True, 
    #                     timeout=35
    #                 )
                    
    #                 response = result.stdout.strip()
    #                 if "Thanks!" in response:
    #                     feedback = "thanks"
    #                 elif "Not now!" in response:
    #                     feedback = "not_now"
    #                 else:
    #                     feedback = "no_response"
                    
    #                 self.button_feedback[nudge_id] = feedback
    #                 self.session_stats[f"{feedback}_count"] += 1
                    
    #                 self.logger.info(f"📊 User feedback for {nudge_id}: {feedback}")
    #                 self._update_satisfaction_multiplier()
                    
    #             except subprocess.TimeoutExpired:
    #                 self.button_feedback[nudge_id] = "no_response"
    #                 self.session_stats["no_response_count"] += 1
    #                 self.logger.info(f"⏱️ No response for {nudge_id} (timeout)")
    #             except Exception as e:
    #                 self.logger.error(f"❌ Error in alert: {e}")
    #                 self.button_feedback[nudge_id] = "no_response"
    #                 self.session_stats["no_response_count"] += 1
            
    #         import threading
    #         thread = threading.Thread(target=show_alert_async, daemon=True)
    #         thread.start()
            
    #         self.logger.info(f"📢 Displayed notification: {title}")
            
    #     except Exception as e:
    #         self.logger.error(f"❌ Error displaying notification: {e}")
