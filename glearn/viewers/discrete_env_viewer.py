import queue
import numpy as np
from .advanced_viewer import AdvancedViewer


class DiscreteEnvViewer(AdvancedViewer):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def prepare(self, trainer):
        super().prepare(trainer)

        self.trainer.add_fetch("Q_values", self.trainer.policy.Q)
        gamma = self.trainer.gamma

        # HACK - more generic way to set these...
        is_frozen = "Frozen" in trainer.env.name
        if is_frozen:
            from gym.envs.toy_text.frozen_lake import LEFT, DOWN, RIGHT, UP
            self.render_Q = True
        else:
            from gym.envs.toy_text.cliffwalking import LEFT, DOWN, RIGHT, UP
            self.render_Q = False
        self.LEFT, self.DOWN, self.RIGHT, self.UP = LEFT, DOWN, RIGHT, UP

        self.optimal_Q, self.optimal_A, self.optimal_V = self._calculate_optimal_QAV(gamma)
        self._add_QAV_maps()

    def _calculate_optimal_QAV(self, gamma):
        # begin with done states
        P = self.env.unwrapped.P
        naction = self.env.unwrapped.action_space.n
        Q = {s: np.full((naction), np.NINF) for s in P.keys()}
        target_rew = {s: np.zeros((naction)) for s in P.keys()}
        fifo = queue.Queue()
        rev_ts = {s: [] for s in P.keys()}
        for s, actions in P.items():
            for action, transitions in actions.items():
                for transition in transitions:
                    prob, ns, rew, done = transition
                    full_t = (s, action, *transition)

                    target_rew[s][action] = rew

                    rev_ts[ns].append(full_t)

                    if done:
                        Q[ns][:] = np.zeros((naction))
                        if ns != s:
                            fifo.put(full_t)

        # calculate optimal Q-values
        while not fifo.empty():
            s, action, prob, ns, rew, done = fifo.get()
            row = int(s / 12)  # HACK
            col = int(s % 12)  # HACK

            previous_states = rev_ts[s]
            if len(previous_states) == 0:
                # print(f"Ignore unreachable state: s=({row},{col}) : {Q[s][action]}")
                continue

            new_Q = target_rew[s][action] + gamma * np.amax(Q[ns])

            if new_Q > Q[s][action]:
                # print(f"Process transition: s=({row},{col}) + a={action} : {Q[s][action]} := {new_Q}")

                Q[s][action] = new_Q
                for rev_t in rev_ts[s]:
                    fifo.put(rev_t)
            # else:
            #     print(f"Maintain transition: s=({row},{col}) + a={action} : {Q[s][action]} >= {new_Q}")

        # calculate optimal actions and V-values
        A = {s: np.argmax(actions) for s, actions in Q.items()}
        V = {s: np.amax(actions) for s, actions in Q.items()}

        return Q, A, V

    def _add_QAV_maps(self):
        # calculate actual Q-map
        naction = self.env.unwrapped.action_space.n
        nrow, ncol = self.env.unwrapped.shape
        optimal_Q_map = np.zeros((nrow, ncol, naction))
        for row in range(nrow):
            for col in range(ncol):
                s = row * ncol + col
                for action in range(naction):
                    optimal_Q_map[row, col, action] = self.optimal_Q[s][action]

        # render optimal AV-map
        self._add_AV_map_label("optimal-AV-map", self.optimal_A, self.optimal_V, x=80, y=10)

        # render optimal Q-map
        if self.render_Q:
            self._add_map_label("optimal-Q-map", optimal_Q_map, x=220, y=10)

    def prepare_render(self):
        # gather environment info
        naction = self.output.size
        nrow, ncol = self.env.unwrapped.shape

        # render ansi env output
        result = self.trainer.env_render_results
        if result is not None:
            self.add_label("map", result, x=10, y=10, width=self.viewer.get_size()[0],
                           multiline=True, font_name="Courier New", font_size=8)

        # build predicted Q-value map
        Q_map = np.zeros((nrow, ncol, naction))
        A = {}
        V = {}
        for row in range(nrow):
            for col in range(ncol):
                # get state representation
                state = row * ncol + col

                # fetch Q-values for state
                results = self.trainer.run("Q_values", {"X": [state]})
                Q_values = results["Q_values"]

                # store Q-values for state
                Q_map[row, col, :] = Q_values
                A[state] = np.argmax(Q_values)
                V[state] = np.amax(Q_values)

        # render AV-map
        self._add_AV_map_label("AV-map", A, V, x=80, y=210)

        # render Q-map
        if self.render_Q:
            self._add_map_label("Q-map", Q_map, x=220, y=210)

    def _add_map_label(self, name, values, x, y):
        if len(values.shape) == 2:
            font_size = 12
            if values.dtype == np.float32 or values.dtype == np.float64:
                values = np.array([[f"{v:.2f}" for v in line] for line in values])
            map_str = "\n".join([" ".join(line) for line in values])
        elif len(values.shape) == 3:
            font_size = 10
            col_width = 16
            map_str = ""
            for line in values:
                qvs = [""] * 3
                for v in line:
                    qvs[0] += f"     {v[self.UP]:.2f}".ljust(col_width)
                    qvs[1] += f"{v[self.LEFT]:.2f}  <>  {v[self.RIGHT]:.2f}".ljust(col_width)
                    qvs[2] += f"     {v[self.DOWN]:.2f}".ljust(col_width)
                map_str += "\n".join(qvs) + "\n\n"

        view_width = self.get_size()[0]
        self.add_label(name, map_str, x=x, y=y, width=view_width,
                       multiline=True, font_name="Courier New", font_size=font_size)

    def _add_AV_map_label(self, name, A, V, x, y):
        action_chars = {self.LEFT: "<", self.DOWN: "v", self.RIGHT: ">", self.UP: "^"}

        nrow, ncol = self.env.unwrapped.shape
        A_map = np.empty((nrow, ncol), dtype=np.object)
        V_map = np.zeros((nrow, ncol))
        for row in range(nrow):
            for col in range(ncol):
                s = row * ncol + col
                A_map[row, col] = action_chars[A[s]]
                V_map[row, col] = V[s]

        AV_map = np.array([[f"{A_map[r, c]}:{V_map[r, c]:.2f}"
                            for c in range(ncol)] for r in range(nrow)])
        self._add_map_label(name, AV_map, x=x, y=y)
