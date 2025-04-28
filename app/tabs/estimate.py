import gradio as gr


class SolarSystemSizing:
    def __init__(self, parent):
        self.parent = parent

        _inputs_fields = {
            "daily_energy_consumption": {
                "label": "Daily Energy Consumption (kWh/day)",
                "value": 10.0,
                "precision": 2,
            },
            "solar_irradiation": {
                "label": "Solar Irradiation (kWh/m²/day)",
                "value": 5.0,
                "precision": 2,
            },
            "module_efficiency": {
                "label": "Module Efficiency (%)",
                "value": 18.0,
                "precision": 1,
            },
            "system_loss": {
                "label": "System Loss (%)",
                "value": 15.0,
                "precision": 1,
            },
            "oversize_factor": {
                "label": "Oversize Factor",
                "value": 1.2,
                "precision": 2,
            },
            "days_of_autonomy": {
                "label": "Days of Autonomy",
                "value": 0,
                "precision": 0,
            },
            "depth_of_discharge": {
                "label": "Depth of Discharge (%)",
                "value": 80.0,
                "precision": 1,
            },
            "panel_power_rating": {
                "label": "Panel Power Rating (W)",
                "value": 400,
                "precision": 0,
            },
        }

        gr.Markdown(
            "Calculate PV array size, number of panels, and optional battery capacity. "
            "Adjust oversize factor to account for seasonal variability."
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Inputs")

                inputs_nb = [
                    gr.Number(**params, interactive=True)
                    for params in _inputs_fields.values()
                ]

                reset_bt = gr.Button("Reset", variant="secondary")

            with gr.Column():
                gr.Markdown("## Results")

                with gr.Row():
                    req_pv_arr_lb = gr.Label(label="Required PV Array (DC)")
                    pan_need_lb = gr.Label(label="Panels Needed")
                    bat_cap_lb = gr.Label(label="Battery Capacity", visible=False)

                submit_bt = gr.Button("Submit", variant="primary")

        self._faq()

        reset_bt.click(
            fn=lambda: [gr.update(value=x["value"]) for x in _inputs_fields.values()],
            outputs=inputs_nb,
        )
        submit_bt.click(
            self._solar_system_sizing,
            inputs_nb,
            [req_pv_arr_lb, pan_need_lb, bat_cap_lb],
        )

        self.parent.select(scroll_to_output=True)

    def _solar_system_sizing(
        self,
        daily_consumption: float,
        irradiation: float,
        panel_efficiency: float = 18.0,
        system_loss: float = 15.0,
        oversize_factor: float = 1.2,
        days_autonomy: float = 0,
        dod: float = 80.0,
        panel_power: float = 400,
    ) -> str:
        """
        Calculate the required solar PV system size, number of panels,
        and optional battery capacity based on user inputs.

        Parameters:
        - daily_consumption: Daily energy need in kWh/day
        - irradiation: Average solar irradiation in kWh/m2/day
        - panel_efficiency: PV module efficiency in %
        - system_loss: Total system loss in % (inverter, wiring, soiling)
        - oversize_factor: Array oversizing factor (e.g., 1.2 for 20% oversize)
        - days_autonomy: Days of battery backup required
        - dod: Depth of discharge of battery in %
        - panel_power: Rated power of single panel in Watts

        Returns:
        - Formatted string with results.
        """
        # Input validation
        errors = []
        if daily_consumption <= 0:
            errors.append("Daily consumption must be positive.")
        if irradiation <= 0:
            errors.append("Irradiation must be positive.")
        if not (0 < panel_efficiency <= 100):
            errors.append("Panel efficiency must be between 0 and 100%.")
        if not (0 <= system_loss < 100):
            errors.append("System loss must be between 0 and 100%.")
        if oversize_factor < 1:
            errors.append("Oversize factor must be >= 1.")
        if days_autonomy < 0:
            errors.append("Days of autonomy cannot be negative.")
        if not (0 < dod <= 100):
            errors.append("Depth of discharge must be between 0 and 100%.")
        if panel_power <= 0:
            errors.append("Panel power rating must be positive.")
        if errors:
            return "Error:\n" + "\n".join(errors)

        # Convert percentages to factors
        loss_factor = 1 - system_loss / 100
        efficiency_factor = panel_efficiency / 100
        dod_factor = dod / 100

        # Calculate required array output before losses
        net_daily_kwh = daily_consumption / loss_factor

        # Calculate required array capacity (DC)
        # Account for irradiation and module efficiency
        required_dc_kw = net_daily_kwh / irradiation / efficiency_factor
        required_dc_kw *= oversize_factor

        # Calculate number of panels
        number_of_panels = (required_dc_kw * 1000) / panel_power

        # Battery sizing (if requested)
        battery_capacity_kwh = 0
        if days_autonomy > 0:
            battery_capacity_kwh = (daily_consumption * days_autonomy) / dod_factor

        # # Build result string
        # result = []
        # result.append(f"Required PV Array (DC): {required_dc_kw:.2f} kW")
        # result.append(f"Panels Needed (@{panel_power}W each): {number_of_panels:.0f}")
        # if battery_capacity_kwh is not None:
        #     result.append(f"Battery Capacity: {battery_capacity_kwh:.2f} kWh")

        # return "\n".join(result)
        return (
            f"{required_dc_kw:.2f} kW",
            f"{number_of_panels:.0f}",
            gr.update(
                value=f"{battery_capacity_kwh:.2f} kWh", visible=days_autonomy > 0
            ),
        )

    def _faq(self):
        gr.Markdown("## FAQ")

        with gr.Accordion("1. Daily Energy Consumption (kWh/day)", open=False):
            gr.Markdown(
                "This is the total amount of electricity you use **per day**.\n\n"
                "**Example**: If your monthly electricity bill says 300 kWh, divide by 30 = **10 kWh/day**.\n\n"
                "**Tip**: Check your utility bill or use your household appliance estimates."
            )

        with gr.Accordion("2. Solar Irradiation (kWh/m²/day)", open=False):
            gr.Markdown(
                "The average amount of sunlight energy available **per square meter per day** at your location.\n\n"
                "**Example**: Typical values are:\n"
                "- Vietnam: 3.6-5.5 kWh/m²/day\n"
                "- Europe: 2.5-5.5 kWh/m²/day\n"
                "- USA (Southwest): 5-7 kWh/m²/day\n\n"
                "**Tip**: Use free online solar maps like [Global Solar Atlas](https://globalsolaratlas.info/map)."
            )

        with gr.Accordion("3. Module Efficiency (%)", open=False):
            gr.Markdown(
                "How efficient the solar panel is at converting sunlight into electricity.\n\n"
                "**Example**: A panel with 20% efficiency turns 20% of sunlight into usable electricity.\n\n"
                "**Typical Range**: 17% to 23% for modern panels."
            )

        with gr.Accordion("4. System Loss (%)", open=False):
            gr.Markdown(
                "Accounts for unavoidable real-world energy losses: inverter losses, wiring resistance, dust, shading, etc.\n\n"
                "**Typical Range**: 10%-20% losses.\n\n"
                "**Tip**: If you don't know, assume 15%."
            )

        with gr.Accordion("5. Oversize Factor", open=False):
            gr.Markdown(
                "A safety margin to make the system slightly bigger to cover cloudy days, seasonal variations, aging panels, and dirt.\n\n"
                "**Example**: 1.2 = design 20% larger than minimum size needed.\n\n"
                "**Tip**: Use 1.1-1.3 for safe sizing."
            )

        with gr.Accordion("6. Days of Autonomy", open=False):
            gr.Markdown(
                "How many days you want to keep your system running **without any sun** (e.g., during storms).\n\n"
                "**Example**: Set 2 if you want 2 days of backup.\n\n"
                "**Tip**: If you don't plan battery backup, leave it at 0."
            )

        with gr.Accordion("7. Depth of Discharge (DoD %)", open=False):
            gr.Markdown(
                "How much of the battery's energy you can use before needing to recharge.\n\n"
                "**Example**:\n"
                "- Lithium batteries: 80%-90% DoD\n"
                "- Lead-acid batteries: 50% DoD\n\n"
                "**Tip**: Higher DoD means better battery usage but depends on battery type."
            )

        with gr.Accordion("8. Panel Power Rating (W)", open=False):
            gr.Markdown(
                "The power output of one solar panel, measured in Watts under standard conditions.\n\n"
                "**Example**: Common panels today are 350W-450W.\n\n"
                "**Tip**: Use the panel's datasheet or label to find this number."
            )

        with gr.Accordion("9. Required PV Array (DC)", open=False):
            gr.Markdown(
                "The total **solar panel array size** you need, in **kiloWatts (kW)**.\n\n"
                "This number tells you how much **total generation capacity** your system must have to meet your daily needs, after accounting for system losses and efficiency."
            )

        with gr.Accordion("10. Panels Needed", open=False):
            gr.Markdown(
                "The **number of solar panels** you need, based on the panel wattage you selected.\n\n"
                "**Example**: If you need 5 kW total and each panel is 400 W, you will need about 13 panels."
            )

        with gr.Accordion("11. Battery Capacity", open=False):
            gr.Markdown(
                "If you requested backup autonomy, this output shows the **battery bank size** needed, in **kWh**.\n\n"
                "It calculates how much stored energy you need to survive the number of days you selected, based on daily consumption and battery depth of discharge."
            )

        with gr.Accordion("12. How is System Size Calculated?", open=False):
            gr.Markdown(r"""
                $$
                \text{System Size (kW)} = \frac{\text{Daily Consumption (kWh/day)}}{\text{Solar Irradiation (kWh/m}^2\text{/day)} \times (1 - \text{System Loss})}
                $$
            """)

        with gr.Accordion("13. How is Number of Panels Calculated?", open=False):
            gr.Markdown(r"""
                $$
                \text{Number of Panels} = \frac{\text{System Size (kW)} \times 1000}{\text{Panel Power Rating (W)}}
                $$
            """)

        with gr.Accordion("14. How is Battery Sizing Calculated?", open=False):
            gr.Markdown(r"""
                $$
                \text{Battery Size (kWh)} = \frac{\text{Daily Consumption (kWh)} \times \text{Days of Autonomy}}{\text{DoD}}
                $$
            """)


class EstimateTab:
    def __init__(self, parent):
        self.parent = parent

    def __call__(self):
        with gr.Tab("Solar System Sizing"):
            SolarSystemSizing(self.parent)
