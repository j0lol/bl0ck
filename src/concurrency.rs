use std::sync::mpsc::Sender;
use std::thread::JoinHandle;

pub struct GameThread<T, U> {
    pub join_handle: JoinHandle<T>,
    pub tx: Sender<U>,
}

impl<T, U> GameThread<T, U> {
    pub fn new(join_handle: JoinHandle<T>, tx: Sender<U>) -> Self {
        GameThread { join_handle, tx }
    }

    // pub fn fulfil(&mut self, join_handle: JoinHandle<T>) {
    //     self.join_handle = Some(join_handle);
    // }
}
