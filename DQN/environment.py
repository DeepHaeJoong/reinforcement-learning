import numpy as np
import pandas as pd

# A task/simulation which needs to be solved by the Agent.
class Environment():
    """
    가전기기 환경
    """

    def __init__(self, state_dim=24):
        # State Space (the energy consumption of time unit)
        self.state_dim = state_dim
        self.AC_states = {i: 0 for i in range(self.state_dim)}
        self.WM_states = {i: 0 for i in range(self.state_dim)}
        self.ESS_states = {i: 2400 if i == 0 else 0 for i in range(self.state_dim)}

        # TOU
        self.TOU = np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.06,
                             0.06, 0.06, 0.12, 0.14, 0.14, 0.12,
                             0.14, 0.14, 0.14, 0.14, 0.12, 0.12,
                             0.12, 0.12, 0.12, 0.12, 0.06, 0.06,
                             ])

        # PV
        self.pv_system = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 250,
                                   500, 1000, 1400, 1750, 2000, 1750,
                                   1400, 1000, 500, 250, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                                   ])

        # outdoor temperature
        self.outdoor_temperature = np.array([23, 23, 23, 23, 24, 24,
                                             27, 28, 28, 31, 31, 32,
                                             33, 33, 32, 32, 30, 30,
                                             29, 27, 27, 26, 25, 24
                                             ])

        # indoor temperature
        self.indoor_temperature = np.zeros(24)
        self.indoor_temperature[-1] = 22  # 실내온도 초기값 (임의로 설정) arbitrary initial value


    def reset(self):
        """ Environment states reset
        Args:
            None
        Returns:
            state
        """

        self.AC_states = {i: 0 for i in range(self.state_dim)}
        self.WM_states = {i: 0 for i in range(self.state_dim)}
        self.ESS_states = {i: 2400 if i == 0 else 0 for i in range(self.state_dim)}

        self.indoor_temperature = np.zeros(24)
        self.indoor_temperature[-1] = 22

        return list(self.AC_states.items())[0], list(self.WM_states.items())[0], list(self.ESS_states.items())[0]


    def AC_step(self, time, action, temp_pref=[23, 25]):
        """
        Args:
            time : 시간
            action : 행동
            temp_pref :선호 온도 [최저, 최대]

        Returns:
            next_state, reward, done
        """
        # the penalty for the consumer thermal discomfort
        kappa = 50

        alpha = 0.8
        beta = -0.02

        self.AC_states[time] = action

        self.indoor_temperature[time] = (self.indoor_temperature[time - 1]+
                                         alpha*(self.outdoor_temperature[time-1]-self.indoor_temperature[time-1])+
                                         beta*action)

        # reward
        if self.indoor_temperature[time] < temp_pref[0]:
            reward = - (self.TOU[time] * action + kappa * (temp_pref[0] - self.indoor_temperature[time]))

        elif self.indoor_temperature[time] > temp_pref[1]:
            reward = - (self.TOU[time] * action + kappa * (self.indoor_temperature[time] - temp_pref[1]))
        else:
            reward = self.TOU[time] * action

        # next_state, done
        if time == 23:
            done = True
            next_state = None
        else:
            done = False
            next_state = list(self.AC_states.items())[time + 1]

        return next_state, reward, done


    def WM_step(self, time, action, WM_pref=[6, 22]):
        """
        Args:
            time : 시간
            action : 행동
            WM_pref :선호 시간 [시작, 종료]

        Returns:
            reward
        """
        # 세타긱 소비전력 (e.g. 500W)
        WM_power = 500

        # penalty
        early_penalty = 50
        late_penalty = 50
        discontinuous_penalty = 100

        start_time = time
        end_time = time + 1

        reward = 0

        if action == 1:

            self.WM_states[time] = WM_power

            # distance = (time - [i for i in [i for i in self.WM_states.values()]].index(500))

            if start_time < WM_pref[0]:
                reward = -(self.TOU[start_time] * WM_power + early_penalty * (WM_pref[0] - start_time))
            elif end_time > WM_pref[1]:
                # reward = -(self.TOU[start_time] * WM_power + late_penalty*(start_time - WM_pref[1]))
                reward = -(self.TOU[start_time] * WM_power + late_penalty * (end_time - WM_pref[1]))
            else:
                reward = -(self.TOU[start_time] * WM_power)

            # next_state, done
            if time == 23:
                done = True
                next_state = None
            else:
                done = False
                next_state = list(self.WM_states.items())[time + 1]

            return next_state, reward, done

        else:

            self.WM_states[time] = 0

            if [i for i in [i for i in self.WM_states.values()]][:time].count(500) > 2:
                reward = - 50
            elif [i for i in [i for i in self.WM_states.values()]][:time].count(500) < 2:
                reward = - 50
            else:
                reward = 0

            # next_state, done
            if time == 23:
                done = True
                next_state = None
            else:
                done = False
                next_state = list(self.WM_states.items())[time + 1]

            return next_state, reward, done

    def ESS_step(self, time, action, min_max_SOE=[800, 4000]):
        """
        Args:
            time : 시간
            action : 행동
            min_max_ESS : minimum, and maximum SOE values [최소, 최대]

        Returns:
            reward
        """
        # the penalties for the ESS overcharging and undercharging
        overcharging_penalty = 50
        undercharging_penalty = 50

        # Q1) ESS 충전 및 사용 시점
        self.ESS_states[time + 1] = (self.ESS_states[time] + self.pv_system[time] + action
                                     - self.AC_states[time] - self.WM_states[time])

        temp_SOE = self.ESS_states[time]

        if temp_SOE > min_max_SOE[1]:
            reward = - (self.TOU[time] * action + overcharging_penalty * (temp_SOE - min_max_SOE[1]))

        elif temp_SOE < min_max_SOE[0]:
            reward = - (self.TOU[time] * action + undercharging_penalty * (min_max_SOE[0] - temp_SOE))

        else:
            reward = self.TOU[time] * action

        # next_state, done
        if time == 23:
            done = True
            next_state = (24, 0)
        else:
            done = False
            next_state = list(self.ESS_states.items())[time + 1]

        return next_state, reward, done