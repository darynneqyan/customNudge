#!/usr/bin/env swift

import Cocoa
import Foundation
import UserNotifications

// Notification category and action identifiers
let NOTIFICATION_CATEGORY = "GUM_NUDGE_CATEGORY"
let ACTION_THANKS = "THANKS_ACTION"
let ACTION_NOT_NOW = "NOT_NOW_ACTION"

class AppDelegate: NSObject, NSApplicationDelegate, UNUserNotificationCenterDelegate {
    var shouldKeepRunning = true
    var selectedAction: String?
    var notificationDelivered = false
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        // This gives us proper app context for notifications
    }
    
    func userNotificationCenter(_ center: UNUserNotificationCenter,
                               didReceive response: UNNotificationResponse,
                               withCompletionHandler completionHandler: @escaping () -> Void) {
        selectedAction = response.actionIdentifier
        shouldKeepRunning = false
        completionHandler()
    }
    
    func userNotificationCenter(_ center: UNUserNotificationCenter,
                               willPresent notification: UNNotification,
                               withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void) {
        // Show notification even when app is in foreground
        completionHandler([.banner, .sound])
        notificationDelivered = true
    }
}

class NotificationManager {
    let delegate = AppDelegate()
    
    func sendNotification(title: String, message: String) {
        // Set up NSApplication for proper app lifecycle
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory) // Background app, no dock icon
        app.delegate = delegate
        
        let center = UNUserNotificationCenter.current()
        center.delegate = delegate
        
        // Check authorization status
        let semaphore = DispatchSemaphore(value: 0)
        var authStatus: UNAuthorizationStatus = .notDetermined
        
        center.getNotificationSettings { settings in
            authStatus = settings.authorizationStatus
            semaphore.signal()
        }
        semaphore.wait()
        
        // Request authorization if needed (will use System Settings permissions)
        if authStatus == .notDetermined {
            let requestSemaphore = DispatchSemaphore(value: 0)
            center.requestAuthorization(options: [.alert, .sound, .badge]) { granted, error in
                if let error = error {
                    let errorMessage = "Authorization error: \(error.localizedDescription)\n"
                    FileHandle.standardError.write(Data(errorMessage.utf8))
                }
                requestSemaphore.signal()
            }
            requestSemaphore.wait()
        } else if authStatus == .denied {
            let errorMessage = "Notifications denied. Enable for 'GUM Notifier' in System Settings.\n"
            FileHandle.standardError.write(Data(errorMessage.utf8))
            print("no_response")
            exit(1)
        }
        
        // Define notification actions
        let thanksAction = UNNotificationAction(
            identifier: ACTION_THANKS,
            title: "Thanks!",
            options: []
        )
        
        let notNowAction = UNNotificationAction(
            identifier: ACTION_NOT_NOW,
            title: "Not now",
            options: []
        )
        
        // Create and register category
        let category = UNNotificationCategory(
            identifier: NOTIFICATION_CATEGORY,
            actions: [thanksAction, notNowAction],
            intentIdentifiers: [],
            options: []
        )
        center.setNotificationCategories([category])
        
        // Create notification content
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = message
        content.sound = UNNotificationSound.default
        content.categoryIdentifier = NOTIFICATION_CATEGORY
        
        // Create and add request
        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: content,
            trigger: nil
        )
        
        let addSemaphore = DispatchSemaphore(value: 0)
        center.add(request) { error in
            if let error = error {
                let errorMessage = "Error adding notification: \(error.localizedDescription)\n"
                FileHandle.standardError.write(Data(errorMessage.utf8))
            }
            addSemaphore.signal()
        }
        addSemaphore.wait()
        
        // Run the app event loop with timeout
        let timeoutDate = Date(timeIntervalSinceNow: 30.0)
        
        // Wait for either user response or timeout
        while delegate.shouldKeepRunning && Date() < timeoutDate {
            // Run the event loop briefly
            let runUntil = Date(timeIntervalSinceNow: 0.1)
            RunLoop.current.run(until: runUntil)
        }
        
        // Output result
        if let action = delegate.selectedAction {
            switch action {
            case ACTION_THANKS:
                print("thanks")
            case ACTION_NOT_NOW:
                print("not_now")
            case UNNotificationDefaultActionIdentifier:
                print("no_response")
            default:
                print("no_response")
            }
        } else {
            print("no_response")
        }
    }
}

func main() {
    // Parse arguments
    let args = CommandLine.arguments
    
    guard args.count >= 3 else {
        let errorMessage = "Usage: \(args[0]) <title> <message>\n"
        FileHandle.standardError.write(Data(errorMessage.utf8))
        exit(1)
    }
    
    let title = args[1]
    let message = args[2]
    
    // Create and run notification manager
    let manager = NotificationManager()
    manager.sendNotification(title: title, message: message)
    
    exit(0)
}

// Start the application
main()
