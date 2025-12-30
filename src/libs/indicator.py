import tkinter as tk
import time


class Indicator(tk.Tk):
    def __init__(self, state_manager: StateManager, geometry: str | None = '20x20+1850+20') -> None:
        super().__init__()

        self.title("Indicator")
        self.geometry(geometry)
        self.state = state_manager
        self.status = self.state.get()
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.attributes('-transparentcolor', 'white')

        self.canvas = tk.Canvas(self, width=20, height=20, bg="black", highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_oval(5, 5, 15, 15, fill="lime")

        self.withdraw()
        self._poll() 

    def render(self):
        self.mainloop()

    def close(self):
        self.destroy()

    def _poll(self):
        self.status = self.state.get()
        if self.status == 'listening':
            self.deiconify()
        
        else:
            self.withdraw()
        
        self.after(1000, self._poll)

if __name__ == "__main__":
    indicator = Indicator(StateManager())
    time.sleep(3)
    indicator.render()