import uuid
import threading
from collections import OrderedDict, deque

class TaskManager:
    def __init__(self, max_task=30):
        self.max_task = max_task
        self.tasks_db = OrderedDict()
        self.tasks_queue = deque()
        self.stop_worker = threading.Event()
        self.lock = threading.Lock()
        # 現在ワーカーが処理中のタスクID（実行中タスクのinterrupt判定に使用）
        self.current_task_id = None
        # interruptを要求されたタスクIDの集合
        self.cancelled_tasks = set()
        self._worker_start()

    def _worker(self):
        while not self.stop_worker.is_set():
            if self.tasks_queue:
                with self.lock:
                    func, args, task_id = self.tasks_queue.popleft()
                    self.current_task_id = task_id
                    # pop直後の同じロック内でin-progressを確定させ、
                    # この間にremoveが来てもpending扱いにならないようにする。
                    if task_id in self.tasks_db:
                        self.tasks_db[task_id]["status"] = "in-progress"
                try:
                    result = func(*args)
                    final_status, final_result = "completed", result
                except Exception as e:
                    final_status, final_result = "failed", str(e)

                # cancelled判定・最終ステータス確定・後片付けを単一クリティカルセクションで
                # 行う。これによりremove側の"interrupted"とworker側の"completed/failed"が
                # 上書き競合するのを防ぎ、finalize後に来たremoveは必ずnot_removableとなる
                # （＝完了済みタスクに対する無駄なinterrupt発火も起きない）。
                with self.lock:
                    if task_id in self.cancelled_tasks:
                        # interrupt要求済みならcompleted/failedに関わらずinterruptedで確定
                        final_status, final_result = "interrupted", None
                    if task_id in self.tasks_db:
                        self.tasks_db[task_id]["status"] = final_status
                        if final_result is not None:
                            self.tasks_db[task_id]["result"] = final_result
                    self.current_task_id = None
                    self.cancelled_tasks.discard(task_id)
            else:
                self.stop_worker.wait(timeout=0.1)

    def _worker_start(self):
        self.worker_thread = threading.Thread(target=self._worker)
        self.worker_thread.start()

    def worker_stop(self):
        self.stop_worker.set()
        self.worker_thread.join()

    def add_task(self, func, *args):
        with self.lock:
            if len(self.tasks_db) >= self.max_task:
                # 最も古いタスクを探す
                oldest_task_id = next(iter(self.tasks_db))
                oldest_task = self.tasks_db[oldest_task_id]
                
                if oldest_task['status'] in ['in-progress', 'pending']:
                    return None, False
                
                self.tasks_db.popitem(last=False)
                self.tasks_queue = deque((f, a, tid) for f, a, tid in self.tasks_queue if tid != oldest_task_id)
    
            task_id = str(uuid.uuid4())
            self.tasks_db[task_id] = {"status": "pending", "result": None}
            self.tasks_queue.append((func, args, task_id))
            
        return task_id, True

    def get_status(self, task_id):
        with self.lock:
            task = self.tasks_db.get(task_id)
            if task:
                if task['status'] == 'pending':
                    pending_tasks = [tid for _, _, tid in self.tasks_queue if self.tasks_db[tid]['status'] == 'pending']
                    queue_position = pending_tasks.index(task_id) + 1 
                    task['queue_position'] = queue_position
                else:
                    task['queue_position'] = None
        return task

    def get_all_tasks(self):
        with self.lock:
            return {k: v["status"] for k, v in self.tasks_db.items()}

    def remove_specific_task(self, task_id, interrupt_callback=None):
        """タスクを削除、または実行中の場合は生成をinterruptする。

        戻り値:
            "removed"       : 待機中(pending)タスクをキューから削除した
            "interrupted"   : 実行中(in-progress)タスクにinterruptを要求した
            "not_found"     : task_idが存在しない
            "not_removable" : 上記以外（既に完了/失敗など）で削除不可
        """
        interrupt_needed = False
        with self.lock:
            task = self.tasks_db.get(task_id)
            if task is None:
                return "not_found"

            status = task["status"]
            if status == "pending":
                del self.tasks_db[task_id]
                self.tasks_queue = deque(item for item in self.tasks_queue if item[2] != task_id)
                return "removed"

            if status == "in-progress" and task_id == self.current_task_id:
                # ワーカーが実行中の生成をinterruptする。
                # ステータスは即座にinterruptedへ更新し、ワーカーは終了時にこれを維持する。
                self.cancelled_tasks.add(task_id)
                task["status"] = "interrupted"
                interrupt_needed = True
            else:
                return "not_removable"

        # interruptのコールバックはグローバルなフラグを立てるだけで非ブロッキングなので
        # ロック外で呼び出す。
        if interrupt_needed and interrupt_callback is not None:
            interrupt_callback()
        return "interrupted"
