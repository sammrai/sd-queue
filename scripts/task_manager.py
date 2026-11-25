import time
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
        # 同じモデルのジョブを続けて処理する（切り替えのたびのモデル読み込み直しを減らす）。
        # 飢餓を防ぐため、連続は max_streak 件まで、先頭の待ちが max_wait 秒を超えたら到着順に戻す
        self.max_streak = 5
        self.max_wait = 900
        # 投入側は前のジョブの完了を受け取ってから次を投げるので、同じモデルの次が届くまで少し待つ
        self.grace = 45
        self.last_finished = 0.0
        self.last_key = None
        self.streak = 0
        self._worker_start()

    @staticmethod
    def _model_key(func, args):
        """ジョブが使うモデルの目印。画像はチェックポイント名、動画は "video"、分からなければ None"""
        req = args[0] if args else None
        settings = getattr(req, "override_settings", None) or {}
        if isinstance(settings, dict) and settings.get("sd_model_checkpoint"):
            return settings["sd_model_checkpoint"]
        if hasattr(req, "duration") and hasattr(req, "image"):
            return "video"
        return None

    def _pop_next(self):
        """次に処理するジョブを取り出す（lock の中で呼ぶ）"""
        head_tid = self.tasks_queue[0][2]
        head_waited = time.time() - self.tasks_db.get(head_tid, {}).get("queued_at", time.time())
        if self.last_key is not None and self.streak < self.max_streak and head_waited < self.max_wait:
            for i, (_, _, tid) in enumerate(self.tasks_queue):
                if self.tasks_db.get(tid, {}).get("model_key") == self.last_key:
                    item = self.tasks_queue[i]
                    del self.tasks_queue[i]
                    return item
            if time.time() - self.last_finished < self.grace:
                return None
        return self.tasks_queue.popleft()

    def _worker(self):
        while not self.stop_worker.is_set():
            with self.lock:
                item = self._pop_next() if self.tasks_queue else None
                if item is not None:
                    func, args, task_id = item
                    key = self.tasks_db.get(task_id, {}).get("model_key")
                    self.streak = self.streak + 1 if key is not None and key == self.last_key else 1
                    self.last_key = key
                    self.current_task_id = task_id
                    # pop直後の同じロック内でin-progressを確定させ、
                    # この間にremoveが来てもpending扱いにならないようにする。
                    if task_id in self.tasks_db:
                        self.tasks_db[task_id]["status"] = "in-progress"
            if item is None:
                self.stop_worker.wait(timeout=0.1)
                continue

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
                self.last_finished = time.time()

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
            self.tasks_db[task_id] = {"status": "pending", "result": None,
                                      "model_key": self._model_key(func, args), "queued_at": time.time()}
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
