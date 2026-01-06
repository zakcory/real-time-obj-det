use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};
use anyhow::{Result};

#[allow(dead_code)]
pub struct FixedSizeQueue<T> {
    queue: Arc<Mutex<VecDeque<T>>>,
    notify: Arc<Notify>,
    capacity: usize,
    on_drop: Option<Arc<dyn Fn(T) + Send + Sync>>,
    pub sender: FixedSizeQueueSender<T>,
    pub receiver: FixedSizeQueueReceiver<T>
}

impl<T> FixedSizeQueue<T> {
    pub fn new<F>(capacity: usize, on_drop: Option<F>) -> Self 
    where
        F: Fn(T) + Send + Sync + 'static
    {
        let queue = Arc::new(Mutex::new(VecDeque::with_capacity(capacity)));
        let notify = Arc::new(Notify::new());
        let on_drop_arc = on_drop.map(|f| Arc::new(f) as Arc<dyn Fn(T) + Send + Sync>);
        
        let sender = FixedSizeQueueSender {
            queue: Arc::clone(&queue),
            notify: Arc::clone(&notify),
            capacity,
            on_drop: on_drop_arc.clone(),
        };
        
        let receiver = FixedSizeQueueReceiver {
            queue: Arc::clone(&queue),
            notify: Arc::clone(&notify)
        };

        Self {
            queue,
            notify,
            capacity,
            on_drop: on_drop_arc,
            sender,
            receiver
        }
    }
}

pub struct FixedSizeQueueSender<T> {
    queue: Arc<Mutex<VecDeque<T>>>,
    notify: Arc<Notify>,
    capacity: usize,
    on_drop: Option<Arc<dyn Fn(T) + Send + Sync>>,
}

impl<T> FixedSizeQueueSender<T> {
    pub fn send_sync(&self, item: T) -> Result<()> {
        // Try to acquire the lock without blocking
        match self.queue.try_lock() {
            Ok(mut queue) => {
                // If at capacity, remove the oldest item (front of queue)
                if queue.len() >= self.capacity {
                    if let Some(dropped_item) = queue.pop_front() {
                        if let Some(ref callback) = self.on_drop {
                            callback(dropped_item);
                        }
                    }
                }
                
                queue.push_back(item);
                drop(queue); // Release lock before notify
                self.notify.notify_one();
                Ok(())
            }
            Err(_) => anyhow::bail!("Queue is full")
        }
    }
    
    // Keep the async version too if you need it elsewhere
    pub async fn send_async(&self, item: T) {
        let mut queue = self.queue.lock().await;
        
        if queue.len() >= self.capacity {
            if let Some(dropped_item) = queue.pop_front() {
                if let Some(ref callback) = self.on_drop {
                    callback(dropped_item);
                }
            }
        }
        
        queue.push_back(item);
        drop(queue);
        self.notify.notify_one();
    }
}

pub struct FixedSizeQueueReceiver<T> {
    queue: Arc<Mutex<VecDeque<T>>>,
    notify: Arc<Notify>,
}

impl<T> FixedSizeQueueReceiver<T> {
    pub async fn recv(&self) -> Option<T> {
        loop {
            let mut queue = self.queue.lock().await;
            if let Some(item) = queue.pop_front() {
                return Some(item);
            }
            
            // Queue is empty, wait for notification
            let notified = self.notify.notified();
            drop(queue); // Release lock before waiting
            notified.await;
        }
    }

    pub async fn drain(&self) -> Vec<T> {
        let mut queue = self.queue.lock().await;
        queue.drain(..).collect()
    }
}