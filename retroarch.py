import gym
import win32gui
from PIL import ImageGrab, Image, ImageFilter
import cv2 as cv
import numpy as np
import time
import re
import keys as k
import hashlib
from windowcapture import WindowCapture
from pytessy.pytessy import PyTessy
from collections import defaultdict
from gym.utils import seeding

wincap = WindowCapture('RetroArch Gearboy 3.3.0')
keys = k.Keys()
ocrReader = PyTessy()

desired_fps = 5
frame_time = 1/desired_fps 


def cellfn(frame):
    cell = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    cell = cv.resize(cell, (11, 8), interpolation = cv.INTER_AREA)
    cell = cell // 32
    return cell

def hashfn(cell):
    return hash(cell.tobytes())

class Weights:
    times_chosen = 0.1
    times_chosen_since_new = 0
    times_seen = 0.3

class Powers:
    times_chosen = 0.5
    times_chosen_since_new = 0.5
    times_seen = 0.5

class Cell(object):
    def __init__(self):
        self.times_chosen = 0
        self.times_chosen_since_new = 0
        self.times_seen = 0

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)
        if key != 'score' and hasattr(self, 'times_seen'):
            self.score = self.cellscore()

    def cntscore(self, a):
        w = getattr(Weights, a)
        p = getattr(Powers, a)
        v = getattr(self, a)
        return w / (v + e1) ** p + e2

    def cellscore(self):
        return self.cntscore('times_chosen')           +\
               self.cntscore('times_chosen_since_new') +\
               self.cntscore('times_seen')             +\
               1

    def visit(self):
        self.times_seen += 1
        return self.times_seen == 1

    def choose(self):
        self.times_chosen += 1
        self.times_chosen_since_new += 1
        return self.ram, self.reward, self.trajectory

e1 = 0.001
e2 = 0.00001

############# Regions Of Interest #############

def roi_1(screen): # Game Over
    roi_1 = screen[219:230, 122:356] # [y1:y2, x1:x2]

    return roi_1

def roi_2(screen): # Lives
    roi_2 = screen[24:41, 171:189] # [y1:y2, x1:x2]

    return roi_2

def roi_3(screen): # Menu ROI for detecting start (START)
    roi_3 = screen[335:355, 146:261] # [y1:y2, x1:x2]

    return roi_3

def roi_4(screen): # Score Numbers
    screen = screen[45:67, 1:143] # [y1:y2, x1:x2]
    gray = get_grayscale(screen)
    resize = cv.resize(gray, (200,75))
    img = opening(resize)
    ocr = Image.fromarray(img)
    ocr = ocr.filter(ImageFilter.SHARPEN)
    imgBytes = ocr.tobytes()
    bytesPerPixel = int(len(imgBytes) / (ocr.width * ocr.height))
    ocr_result = ocrReader.read(ocr.tobytes(), ocr.width, ocr.height, bytesPerPixel, raw=True, resolution=600)
    result = str(ocr_result)
    roi_4 = replace_chars(result)[0]

    return roi_4

def get_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv.morphologyEx(image, cv.MORPH_OPEN, kernel)

def replace_chars(text):
    """
    Replaces all characters instead of numbers from 'text'.
    
    :param text: Text string to be filtered
    :return: Resulting number
    """
    list_of_numbers = re.findall(r'\d+', text)
    #result_number = ''.join(list_of_numbers)
    return list_of_numbers

class RetroArch(gym.Env):
    def __init__(self):
        super().__init__()
        self.window_title = 'RetroArch Gearboy 3.3.0'
        self.action_space = gym.spaces.Discrete(18)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(433, 476, 3), dtype=np.uint8)
        self.highscore = 0
        self.last_time = 0

        self.frames = 0  
        self.trajectory = []
        self.iterations = 0
        self.found_new_cell = False
        self.game_score = 0
        self.time_left = 0
        self.fps = 0

        self.max_reward = 0
        self.counter = 0

        self.seed()

    def seed(self, seed = None):
        """
        Seeds both the internal numpy rng for stochastic frame skip
        as well as the ALE RNG.

        This function must also initialize the ROM and set the corresponding
        mode and difficulty. `seed` may be called to initialize the environment
        during deserialization by Gym so these side-effects must reside here.

        Args:
            seed: int => Manually set the seed for RNG.
        Returns:
            tuple[int, int] => (np seed, ALE seed)
        """
        ss = np.random.SeedSequence(seed)
        seed1, seed2 = ss.generate_state(n_words=2)

        self.np_random = np.random.default_rng(seed1)

        return (
            seed1,
            seed2,
        )

    def step(self, action):
        
        start_time = time.time()
        # Implement your environment's logic here
        observation = wincap.get_screenshot()
        self.frames += 1
        reward = 0
        done = False

        self.trajectory.append(action)

        ### SCORE ###
        try:
            score = roi_4(observation)
            score = int(score)
        except:
            score = 0

        if score > self.highscore:
            self.highscore = score
            reward = score / 100

        if score > self.max_reward:
            self.max_reward = score
            self.counter = 0
        else:
            self.counter += 1

        ### LIVES LOST ###
        lives = roi_2(observation)
        lives_light_pix = np.sum(lives >= 50)
        #print('lives_light_pix', lives_light_pix)
        lives_dark_pix = np.sum(lives <= 49)
        #print('lives_dark_pix', lives_dark_pix)
        '''
        if lives_light_pix == 489 and lives_dark_pix == 429: # 1 Life Left
            reward += - 1000
        '''
        if lives_light_pix == 288 and lives_dark_pix == 360: # 0 Lives Left
            reward += - 1000
            done = True

        ### GAME OVER ###
        game = roi_1(observation)
        number_of_light_pix = np.sum(game >= 50)
        number_of_dark_pix = np.sum(game <= 49)
        if number_of_light_pix == 4176 and number_of_dark_pix == 3546 or self.counter == 1000:
            done = True
        else:
            cell = cellfn(observation)
            cellhash = hashfn(cell)
            cell = self.archive[cellhash]
            first_visit = cell.visit()
            if first_visit or score > cell.reward or score == cell.reward and len(self.trajectory) < len(cell.trajectory):
                cell.ram = cellhash
                cell.reward = score
                cell.trajectory = self.trajectory.copy()
                cell.times_chosen = 0
                cell.times_chosen_since_new = 0
                found_new_cell = True
                reward += 0.1

        # Actions
        if action == 0:
            keys.directKey("RIGHT", keys.key_release)
            keys.directKey("LEFT", keys.key_release)
            keys.directKey("DOWN", keys.key_release)
            keys.directKey("UP", keys.key_release)
            keys.directKey("X", keys.key_release)
            keys.directKey("Z", keys.key_release)
        elif action == 1:
            keys.directKey("RIGHT")
        elif action == 2:
            keys.directKey("LEFT")
        elif action == 3:
            keys.directKey("DOWN")
        elif action == 4:
            keys.directKey("UP")
        elif action == 5:
            keys.directKey("X")
        elif action == 6:
            keys.directKey("Z")
        elif action == 7:
            keys.directKey("X")
            keys.directKey("Z")
        elif action == 8:
            keys.directKey("UP")
            keys.directKey("LEFT")
        elif action == 9:
            keys.directKey("UP")
            keys.directKey("RIGHT")
        elif action == 10:
            keys.directKey("DOWN")
            keys.directKey("LEFT")
        elif action == 11:
            keys.directKey("DOWN")
            keys.directKey("RIGHT")
        elif action == 12:
            keys.directKey("X")
            keys.directKey("LEFT")
        elif action == 13:
            keys.directKey("X")
            keys.directKey("RIGHT")
        elif action == 14:
            keys.directKey("X")
            keys.directKey("UP")
        elif action == 15:
            keys.directKey("X")
            keys.directKey("DOWN")
        elif action == 16:
            keys.directKey("Z")
            keys.directKey("LEFT")
        elif action == 17:
            keys.directKey("Z")
            keys.directKey("RIGHT")

        info = {}
        reward += -0.01 * 1
        #print('reward', reward)

        elapsed = time.time() - start_time
        
        #print("Game FPS: ", fps)
        time.sleep(max(frame_time - elapsed, 0))
        new_fps = time.time() - start_time
        self.fps = 1 / new_fps

        return observation, reward, done, info

    def reset(self):

        self.iterations += 1
        print ("Iterations: %d, Frames: %d, Max Reward: %d, Game FPS: %d" % (self.iterations, self.frames, self.highscore, self.fps))

        '''
        if self.found_new_cell and self.iteration > 0:
            restore_cell.times_chosen_since_new = 0
        
        try:
            self.iteration += 1
            scores = np.array([cell.score for cell in self.archive.values()])
            hashes = [*self.archive]
            probs = scores / scores.sum()
            restore = np.random.choice(hashes, p = probs)
            restore_cell = self.archive[restore]
            #ram, score, trajectory = restore_cell.choose()
        except:
            pass
        '''

        # Restart RetroArch game
        keys.directKey("H")
        time.sleep(0.1)
        keys.directKey("H", keys.key_release)
        time.sleep(0.5)
        keys.directKey("RETURN")
        time.sleep(0.1)
        keys.directKey("RETURN", keys.key_release)
        time.sleep(0.1)

        self.highscore = 0
        self.time_left = 0
        self.max_reward = 0
        self.counter = 0

        self.archive = defaultdict(lambda: Cell())  
        self.trajectory = []
        self.found_new_cell = False

        observation = wincap.get_screenshot()

        return observation

    def render(self, mode='human'):
        pass

    def get_action_meanings(self):

        return {0: "NOOP",
                1: "DPAD_RIGHT",
                2: "DPAD_LEFT",
                3: "DPAD_DOWN",
                4: "DPAD_UP",
                5: "A",
                6: "B",
                7: "AB",
                8: "DPAD_UPLEFT",
                9: "DPAD_UPRIGHT",
                10: "DPAD_DOWNLEFT",
                11: "DPAD_DOWNRIGHT",
                12: "A_LEFT",
                13: "A_RIGHT",
                14: "A_UP",
                15: "A_DOWN",
                16: "B_LEFT",
                17: "B_RIGHT"
                }