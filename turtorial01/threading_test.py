#codeing = utf8
import threading, time, random

count = 0
class MyThread(threading.Thread):

    def __init__(self, lock, threadName):
        super(MyThread, self).__init__(name = threadName)
        self.lock = lock

    def run(self):
        global count
        self.lock.acquire()
        for i in range(100):
            count += 1
            time.sleep(0.3)
            print(self.name, count)
        self.lock.release()

def doWaiting():
    print('start waiting:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    time.sleep(3)
    print('end waiting:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    thread1 = threading.Thread(target = doWaiting)
    thread1.start()
    time.sleep(1)
    print('start join')
    thread1.join()
    print('end join')


lock = threading.Lock()
for i in range(2):
    MyThread(lock, "MyThreadName:" + str(i)).start()
doWaiting()