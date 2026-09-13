import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import main

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class F1PredictorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("F1 Telemetry & Degradation Predictor")
        self.geometry("1000x750")

        # Layout Configuration
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Control Panel Frame
        self.ctrl_frame = ctk.CTkFrame(self)
        self.ctrl_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        # Year Dropdown
        self.lbl_year = ctk.CTkLabel(self.ctrl_frame, text="Year:", font=ctk.CTkFont(weight="bold"))
        self.lbl_year.pack(side="left", padx=(15, 5), pady=15)

        years = [str(y) for y in range(2020, 1949, -1)]  # 2020 down to 1950
        self.combo_year = ctk.CTkOptionMenu(self.ctrl_frame, values=years, command=self.on_year_change)
        self.combo_year.set("2019")
        self.combo_year.pack(side="left", padx=5, pady=15)

        # Race Map Dropdown
        self.lbl_race = ctk.CTkLabel(self.ctrl_frame, text="Grand Prix:", font=ctk.CTkFont(weight="bold"))
        self.lbl_race.pack(side="left", padx=(15, 5), pady=15)

        self.combo_race = ctk.CTkOptionMenu(self.ctrl_frame, values=["Loading..."])
        self.combo_race.pack(side="left", padx=5, pady=15)

        # Run Pipeline Button
        self.btn_run = ctk.CTkButton(
            self.ctrl_frame, 
            text="Run ML Pipeline", 
            font=ctk.CTkFont(weight="bold"),
            command=self.run_pipeline
        )
        self.btn_run.pack(side="right", padx=15, pady=15)

        # Main Workspace Tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")

        self.tab_before = self.tabview.add("1. Raw Model (Before)")
        self.tab_after = self.tabview.add("2. Fuel-Corrected Model (After)")

        # Status / Error Label
        self.status_label = ctk.CTkLabel(
            self.tab_before, 
            text="Initializing F1 Dataset...", 
            font=ctk.CTkFont(size=14)
        )
        self.status_label.pack(expand=True, padx=20, pady=20)

        # State storage
        self.canvas_before = None
        self.canvas_after = None
        self.race_df = None
        self.results_df = None
        self.lap_times_df = None
        self.pit_stops_df = None

        # Async Data Loading Initialization
        self.after(100, self.init_dataset)

    def init_dataset(self):
        try:
            self.race_df, self.results_df, self.lap_times_df, self.pit_stops_df = main.get_datasets()
            self.on_year_change("2019")
            self.combo_race.set("Spanish Grand Prix")
            self.status_label.configure(text="Dataset loaded. Select a year and Grand Prix, then click 'Run ML Pipeline'.")
        except Exception as e:
            self.status_label.configure(text=f"Error loading Kaggle Dataset: {str(e)}", text_color="red")

    def on_year_change(self, selected_year):
        if self.race_df is not None:
            available_races = main.get_available_races(self.race_df, selected_year)
            if available_races:
                self.combo_race.configure(values=available_races)
                self.combo_race.set(available_races[0])
            else:
                self.combo_race.configure(values=["No Races Found"])
                self.combo_race.set("No Races Found")

    def clear_canvases(self):
        if self.canvas_before:
            self.canvas_before.get_tk_widget().destroy()
            self.canvas_before = None
        if self.canvas_after:
            self.canvas_after.get_tk_widget().destroy()
            self.canvas_after = None

    def run_pipeline(self):
        selected_year = self.combo_year.get()
        selected_race = self.combo_race.get()

        # Re-create status label if destroyed
        if not self.status_label.winfo_exists():
            self.status_label = ctk.CTkLabel(self.tab_before, font=ctk.CTkFont(size=14))
            self.status_label.pack(expand=True, padx=20, pady=20)

        self.clear_canvases()
        self.status_label.configure(text=f"Processing telemetry for {selected_year} {selected_race}...", text_color="white")
        self.update_idletasks()

        # Direct Error Handling Block for GUI
        try:
            train_df, test_df = main.load_and_preprocess_data(
                self.race_df, self.results_df, self.lap_times_df, self.pit_stops_df, 
                selected_year, selected_race
            )
            
            fig_before = main.generate_before_plot(train_df, test_df, selected_race, selected_year)
            fig_after = main.generate_after_plot(train_df, test_df, selected_race, selected_year)

            # Hide status label once plots render successfully
            self.status_label.pack_forget()

            # Display "Before" plot in Tab 1
            self.canvas_before = FigureCanvasTkAgg(fig_before, master=self.tab_before)
            self.canvas_before.draw()
            self.canvas_before.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

            # Display "After" plot in Tab 2
            self.canvas_after = FigureCanvasTkAgg(fig_after, master=self.tab_after)
            self.canvas_after.draw()
            self.canvas_after.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        except ValueError as val_err:
            # Display graceful in-app error for invalid/incomplete telemetry
            self.status_label.pack(expand=True, padx=20, pady=20)
            self.status_label.configure(
                text=f"⚠️ Telemetry Error:\n{str(val_err)}", 
                text_color="#FF5555"
            )
        except Exception as e:
            # Display general processing error
            self.status_label.pack(expand=True, padx=20, pady=20)
            self.status_label.configure(
                text=f"❌ Failed to process race:\n{str(e)}", 
                text_color="#FF5555"
            )

if __name__ == "__main__":
    app = F1PredictorApp()
    app.mainloop()