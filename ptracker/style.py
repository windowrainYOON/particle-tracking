"""matplotlib 한글 글꼴과 ffmpeg 경로 설정 (macOS / Windows / Linux 공통)."""
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_KOREAN_FONTS = ["AppleGothic", "Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans KR", "NanumGothic",
                 "Malgun Gothic", "Noto Sans CJK JP", "DejaVu Sans"]
_done = False


def setup():
    global _done
    if _done:
        return
    avail = {f.name for f in font_manager.fontManager.ttflist}
    fams = [f for f in _KOREAN_FONTS if f in avail] or ["DejaVu Sans"]
    plt.rcParams["font.family"] = fams
    plt.rcParams["axes.unicode_minus"] = False
    ff = None
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ff = shutil.which("ffmpeg")
    if ff:
        plt.rcParams["animation.ffmpeg_path"] = ff
    _done = True


def movie_writer(fps=3, bitrate=5000):
    """(writer, 확장자). ffmpeg가 없으면 GIF로 대체."""
    from matplotlib import animation
    setup()
    if animation.FFMpegWriter.isAvailable():
        return animation.FFMpegWriter(fps=fps, bitrate=bitrate), ".mp4"
    return animation.PillowWriter(fps=fps), ".gif"
