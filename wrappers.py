"""

"""

import gymnasium as gym
import numpy as np

# TODO: Remove this if its not being used
# class ImgObsPositionWrapper(gym.ObservationWrapper):
#     """
#     Use the image as the only observation output, no language/mission.
#     """

#     def __init__(self, env):
#         """A wrapper that makes image the only observation.

#         Args:
#             env: The environment to apply the wrapper
#         """
#         super().__init__(env)
#         self.observation_space = env.observation_space.spaces["image"]

#     def observation(self, obs):
#         return obs["image"], obs
    

class PartialAndTotalRecordVideo(gym.wrappers.RecordVideo):
    def _capture_frame(self):
        assert self.recording, "Cannot capture a frame, recording wasn't started."

        frame = self.render()
        if isinstance(frame, list):
            if len(frame) == 0:  # render was called
                return
            self.render_history += frame
            frame = frame[-1]

        if isinstance(frame, np.ndarray):
            self.recorded_frames.append(frame)
        else:
            self.stop_recording()
            logger.warn(
                f"Recording stopped: expected type of frame returned by render to be a numpy array, got instead {type(frame)}."
            )

    @property
    def enabled(self):
        return self.recording

    def render(self):
        img_total, img_partial = self.env.unwrapped.render_path_visualizations()
        # img_partial has 1 channel; repeat to make it 3 channels
        # Handles the fact that the partial view is grayscale
        if img_partial.ndim == 3 and img_partial.shape[2] == 1:
            img_partial = np.repeat(img_partial, 3, axis=2)

        # INSERT_YOUR_CODE
        # 8x the size of img_total and img_partial using nearest neighbor upsampling
        def upsample(img, scale):
            h, w, c = img.shape
            return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

        img_total = upsample(img_total, 8)
        img_partial = upsample(img_partial, 8)
        # Stack img_total and img_partial vertically with a padding in between

        # Ensure both images have the same width; if not, pad the smaller one
        h1, w1, c1 = img_total.shape
        h2, w2, c2 = img_partial.shape
        pad_color = 255  # white padding

        # Determine the max width and number of channels
        max_w = max(w1, w2)
        max_c = max(c1, c2)

        def pad_img(img, target_w, target_c):
            h, w, c = img.shape
            # Pad width if needed, split between left and right
            if w < target_w:
                total_pad = target_w - w
                pad_left = total_pad // 2
                pad_right = total_pad - pad_left
                pad_width = ((0, 0), (pad_left, pad_right), (0, 0))
                img = np.pad(img, pad_width, mode="constant", constant_values=pad_color)
            # Pad channels if needed
            if c < target_c:
                total_pad = target_c - c
                pad_left = total_pad // 2
                pad_right = total_pad - pad_left
                pad_channels = ((0, 0), (0, 0), (pad_left, pad_right))
                img = np.pad(
                    img, pad_channels, mode="constant", constant_values=pad_color
                )
            return img

        img_total_padded = pad_img(img_total, max_w, max_c)
        img_partial_padded = pad_img(img_partial, max_w, max_c)

        # Create a padding row (e.g., 10 pixels high)
        pad_height = 10
        padding = np.full((pad_height, max_w, max_c), pad_color, dtype=img_total.dtype)
        padding_bottom = np.full((30, max_w, max_c), pad_color, dtype=img_total.dtype)

        # Concatenate: total on top, then padding, then partial
        render_out = [
            np.concatenate(
                [img_total_padded, padding, img_partial_padded, padding_bottom], axis=0
            )
        ]
        if self.recording and isinstance(render_out, list):
            self.recorded_frames += render_out

        if len(self.render_history) > 0:
            tmp_history = self.render_history
            self.render_history = []
            return tmp_history + render_out
        else:
            return render_out