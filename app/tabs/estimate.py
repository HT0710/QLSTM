import gradio as gr


class ROI:
    def __init__(self, parent):
        self.parent = parent

        _inputs_fields = {
            "system_cost": {
                "label": "Initial System Cost",
                "value": 10_000,
                "precision": 2,
            },
            "annual_output": {
                "label": "Annual Energy Output (kWh)",
                "value": 5_000,
                "precision": 0,
            },
            "electricity_rate": {
                "label": "Electricity Rate (per kWh)",
                "value": 0.20,
                "precision": 3,
            },
            "system_lifetime": {
                "label": "System Lifetime (years)",
                "value": 25,
                "precision": 0,
            },
        }

        gr.Markdown(
            "Estimate financial return based on installation cost, energy savings, incentives, and system lifespan. "
            "Helps assess long-term economic benefits of your solar investment."
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

                self.roi_lb = gr.Label(label="ROI (%)")

                self.calc_roi = gr.Button("Calculate", variant="primary")

        self._faq()
        reset_bt.click(
            fn=lambda: [gr.update(value=x["value"]) for x in _inputs_fields.values()],
            outputs=inputs_nb,
        )
        self.calc_roi.click(
            fn=self._calculate_roi,
            inputs=inputs_nb,
            outputs=[self.roi_lb],
        )

    def _calculate_roi(self, cost, output, rate, lifetime):
        # net savings = total revenue - cost
        total_savings = output * rate * lifetime
        roi = (total_savings - cost) / cost * 100
        return f"{roi:.1f}"

    def _faq(self):
        gr.Markdown("## FAQ")
        with gr.Accordion("1. Initial System Cost", open=False):
            gr.Markdown(
                "The total upfront investment including panels, inverters, mounting, and installation fees.\n\n"
                "**Example**: A 5 kW system might cost around $10,000.\n\n"
                "**Tip**: Get multiple quotes and include all soft costs (permits, labor)."
            )
        with gr.Accordion("2. Annual Energy Output (kWh)", open=False):
            gr.Markdown(
                "The estimated electricity your system will generate per year.\n\n"
                "**Example**: 5,000 kWh/year for a typical residential PV system.\n\n"
                "**Tip**: Use your Energy Yield tab or utility bills."
            )
        with gr.Accordion("3. Electricity Rate (per kWh)", open=False):
            gr.Markdown(
                "Your current cost of grid electricity.\n\n"
                "**Example**: $0.20/kWh if your bill is $200 for 1,000 kWh.\n\n"
                "**Tip**: Check recent utility bills; rates can vary seasonally."
            )
        with gr.Accordion("4. System Lifetime (years)", open=False):
            gr.Markdown(
                "The expected operational lifespan of your PV system.\n\n"
                "**Typical Range**: 20-30 years.\n\n"
                "**Tip**: Manufacturer warranty often indicates reliable lifetime."
            )
        with gr.Accordion("5. How is ROI (%) Calculated?", open=False):
            gr.Markdown(
                "**Formula**:"
                "$$ROI = \\frac{(\\text{Annual Output} \\times \\text{Rate} \\times \\text{Lifetime}) - \\text{Initial Cost}}{\\text{Initial Cost}} \\times 100$$\n\n"
                "It measures the percentage return on your investment over the system life."
            )


class Payback:
    def __init__(self, parent):
        self.parent = parent

        _inputs_fields = {
            "system_cost": {
                "label": "Initial System Cost",
                "value": 10_000,
                "precision": 2,
            },
            "annual_output": {
                "label": "Annual Energy Output (kWh)",
                "value": 5_000,
                "precision": 0,
            },
            "electricity_rate": {
                "label": "Electricity Rate (per kWh)",
                "value": 0.20,
                "precision": 3,
            },
        }

        gr.Markdown(
            "Calculate the number of years required to recover the initial investment from energy savings. "
            "Useful for comparing system cost-efficiency over time."
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

                self.payback_lb = gr.Label(label="Payback Period (years)")

                self.calc_payback = gr.Button("Calculate", variant="primary")

        self._faq()
        reset_bt.click(
            fn=lambda: [gr.update(value=x["value"]) for x in _inputs_fields.values()],
            outputs=inputs_nb,
        )
        self.calc_payback.click(
            fn=self._calculate_payback,
            inputs=inputs_nb,
            outputs=[self.payback_lb],
        )

    def _calculate_payback(self, cost, output, rate):
        annual_savings = output * rate
        if annual_savings <= 0:
            return "Error: Savings must be positive."
        payback = cost / annual_savings
        return f"{payback:.1f}"

    def _faq(self):
        gr.Markdown("## FAQ")

        with gr.Accordion("1. Initial System Cost", open=False):
            gr.Markdown(
                "The total upfront investment including panels, inverters, mounting, and installation fees.\n\n"
                "**Example**: A 5 kW system might cost around $10,000.\n\n"
                "**Tip**: Get multiple quotes and include all soft costs (permits, labor)."
            )
        with gr.Accordion("2. Annual Energy Output (kWh)", open=False):
            gr.Markdown(
                "The estimated electricity your system will generate per year.\n\n"
                "**Example**: 5,000 kWh/year for a typical residential PV system.\n\n"
                "**Tip**: Use your Energy Yield tab or utility bills."
            )
        with gr.Accordion("3. Electricity Rate (per kWh)", open=False):
            gr.Markdown(
                "Your current cost of grid electricity.\n\n"
                "**Example**: $0.20/kWh if your bill is $200 for 1,000 kWh.\n\n"
                "**Tip**: Check recent utility bills; rates can vary seasonally."
            )
        with gr.Accordion("4. How is Payback Period (years) Calculated?", open=False):
            gr.Markdown(
                "**Formula**:"
                "$$\\text{Payback Period} = \\frac{\\text{Initial Cost}}{\\text{Annual Savings}}$$\n\n"
                "It shows how many years until your investment is recovered."
            )


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

                calc_bt = gr.Button("Calculate", variant="primary")

        self._faq()

        reset_bt.click(
            fn=lambda: [gr.update(value=x["value"]) for x in _inputs_fields.values()],
            outputs=inputs_nb,
        )
        calc_bt.click(
            self._solar_system_sizing,
            inputs_nb,
            [req_pv_arr_lb, pan_need_lb, bat_cap_lb],
        )

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
                "**Example**:\n"
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

        with gr.Accordion("9. How is Required PV Array (DC) Calculated?", open=False):
            gr.Markdown(
                "The total **solar panel array size** you need, in **kilowatt-peak (kWp)**, which is the rated output of the panels under standard conditions (STC).\n\n"
                "**Formula**:"
                "$$\\text{Required PV Array} = \\frac{\\text{Daily Consumption}}{\\text{Solar Irradiation} \\times (1 - \\text{System Loss})}$$\n\n"
                "This number tells you how much DC-rated total generation capacity (in kWp) your solar array must have to meet your daily energy needs, after accounting for system losses and average solar resource."
            )

        with gr.Accordion("10. How is Panels Needed Calculated?", open=False):
            gr.Markdown(
                "The **number of solar panels** you need, based on the panel wattage you selected.\n\n"
                "**Formula**:"
                "$$\\text{Number of Panels} = \\frac{\\text{System Size} \\times 1000}{\\text{Panel Power Rating}}$$\n\n"
                "**Example**: If you need 5 kW total and each panel is 400 W, you will need about 13 panels."
            )

        with gr.Accordion("11. How is Battery Capacity (kWh) Calculated?", open=False):
            gr.Markdown(
                "If you requested backup autonomy, this output shows the **battery bank size** needed, in **kWh**.\n\n"
                "**Formula**:"
                "$$\\text{Battery Capacity} = \\frac{\\text{Daily Consumption} \\times \\text{Days of Autonomy}}{\\text{DoD}}$$\n\n"
                "It calculates how much stored energy you need to survive the number of days you selected, based on daily consumption and battery depth of discharge."
            )


class EnergyYield:
    def __init__(self, parent):
        self.parent = parent

        _inputs_fields = {
            "annual_irradiation": {
                "label": "Annual Irradiation (kWh/m²/year)",
                "value": 1_500,
                "precision": 0,
            },
            "installed_capacity": {
                "label": "Installed Capacity (kWp)",
                "value": 5.0,
                "precision": 2,
            },
            "performance_ratio": {
                "label": "Performance Ratio (%)",
                "value": 75.0,
                "precision": 1,
            },
        }

        gr.Markdown(
            "Estimate annual energy production and capacity factor based on system size, solar resource, and real-world performance."
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
                    self.aep_lb = gr.Label(label="Annual Energy Production (kWh)")
                    self.cf_lb = gr.Label(label="Capacity Factor (%)")

                self.calc_bt = gr.Button("Calculate", variant="primary")

        self._faq()

        reset_bt.click(
            fn=lambda: [gr.update(value=x["value"]) for x in _inputs_fields.values()],
            outputs=inputs_nb,
        )
        self.calc_bt.click(self._estimate_yield, inputs_nb, [self.aep_lb, self.cf_lb])

    def _estimate_yield(self, annual_irrad: float, capacity_kwp: float, pr_pct: float):
        """
        AEP = capacity * irradiation * (PR/100)
        CF = AEP / (capacity * 8760h) * 100%
        """
        # validation
        errors = []
        if annual_irrad <= 0:
            errors.append("Irradiation must be positive.")
        if capacity_kwp <= 0:
            errors.append("Capacity must be positive.")
        if not (0 < pr_pct <= 100):
            errors.append("PR must be 0-100%.")
        if errors:
            return "\n".join(errors), ""

        pr_factor = pr_pct / 100
        aep = capacity_kwp * annual_irrad * pr_factor
        cf = (aep / (capacity_kwp * 8760)) * 100

        return f"{aep:,.0f}", f"{cf:.1f}"

    def _faq(self):
        gr.Markdown("## FAQ")

        with gr.Accordion("1. Annual Irradiation (kWh/m²/year)", open=False):
            gr.Markdown(
                "This represents the **total solar energy** received per square meter over a year at your location.\n\n"
                "**Typical Values**:\n"
                "- Southern Vietnam: ~1000-2000\n"
                "- Northern Vietnam: ~500-1000\n"
                "- Southern Europe: ~1600-2000\n"
                "- Desert regions: ~2200-2400+\n\n"
                "**Tip**: Use satellite tools like [Global Solar Atlas](https://globalsolaratlas.info/map) to get the value for your site."
            )

        with gr.Accordion("2. Installed Capacity (kWp)", open=False):
            gr.Markdown(
                "This is the **total rated power** of your solar PV system under standard test conditions, in kilowatts peak (kWp).\n\n"
                "Example: If you install 12 panels rated at 400W each, your installed capacity is:\n"
                "12 x 400W = 4.8 kWp."
            )

        with gr.Accordion("3. Performance Ratio (%)", open=False):
            gr.Markdown(
                "The **Performance Ratio (PR)** quantifies real-world system efficiency after accounting for various losses like inverter inefficiency, temperature, soiling, shading, and wiring.\n\n"
                "**Typical Range**: 70%-85% depending on climate and equipment quality.\n\n"
                "**Formula**:"
                "$$PR = \\frac{\\text{Actual Energy Output}}{\\text{Theoretical Energy Output}}$$\n\n"
                "A higher PR means your system performs closer to its theoretical maximum."
            )

        with gr.Accordion(
            "4. How is Annual Energy Production (kWh) Calculated?", open=False
        ):
            gr.Markdown(
                "This is the estimated **total amount of electricity (AEP)** (in kWh) your system will produce in one year, considering solar resource and losses.\n\n"
                "**Formula**:"
                "$$AEP = \\text{Installed Capacity} \\times \\text{Annual Irradiation} \\times \\frac{PR}{100}$$\n\n"
                "It tells you how much energy you can expect to generate from your system per year."
            )

        with gr.Accordion("5. How is Capacity Factor (%) Calculated?", open=False):
            gr.Markdown(
                "The **Capacity Factor (CF)** represents how efficiently a power plant operates compared to its maximum possible output over time.\n\n"
                "For solar systems, it accounts for night, cloudy weather, and system losses.\n\n"
                "**Typical Range**:\n"
                "- Poor site: 10-12%\n"
                "- Good site: 15-18%\n"
                "- Excellent site: 20-25%\n\n"
                "**Formula**:"
                "$$CF = \\frac{\\text{AEP}}{\\text{Installed Capacity} \\times 8760 \\text{ hours}} \\times 100\\%$$\n\n"
                "Higher CF means more consistent and productive generation."
            )


class EnvironmentalImpact:
    def __init__(self, parent):
        self.parent = parent

        _inputs_fields = {
            "annual_output": {
                "label": "Annual Energy Output (kWh)",
                "value": 5_000,
                "precision": 0,
            },
            "grid_emission": {
                "label": "Grid Emission Factor (kg CO₂/kWh)",
                "value": 0.5,
                "precision": 3,
            },
            "embodied_energy": {
                "label": "Embodied Energy (kWh)",
                "value": 15_000,
                "precision": 0,
            },
            "lifecycle_emissions": {
                "label": "Lifecycle Emissions (kg CO₂-eq)",
                "value": 10_000,
                "precision": 0,
            },
            "system_lifetime": {
                "label": "System Lifetime (years)",
                "value": 25,
                "precision": 0,
            },
        }

        gr.Markdown(
            "Quantify carbon emissions offset, equivalent trees planted, or fossil fuel displacement. "
            "Helps understand the sustainability benefits of solar adoption."
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
                    self.co2_lb = gr.Label(label="CO₂ Avoided (tons/year)")
                    self.epbt_lb = gr.Label(label="Energy Payback Time (years)")
                    self.ghg_lb = gr.Label(label="GHG Intensity (g CO₂-eq/kWh)")

                self.calc_impact = gr.Button("Calculate", variant="primary")

        self._faq()
        reset_bt.click(
            fn=lambda: [gr.update(value=x["value"]) for x in _inputs_fields.values()],
            outputs=inputs_nb,
        )
        self.calc_impact.click(
            fn=self._calculate_environmental,
            inputs=inputs_nb,
            outputs=[self.co2_lb, self.epbt_lb, self.ghg_lb],
        )

    def _calculate_environmental(self, output, factor, embodied, lifecycle, lifetime):
        # CO2 avoided
        co2_avoided = output * factor / 1000  # tons/year
        # EPBT
        epbt = embodied / output if output > 0 else 0
        # GHG intensity
        total_prod = output * lifetime  # assume 25-year lifetime
        ghg_intensity = lifecycle * 1000 / total_prod if total_prod > 0 else 0  # g/kWh
        return (f"{co2_avoided:.2f}", f"{epbt:.2f}", f"{ghg_intensity:.1f}")

    def _faq(self):
        gr.Markdown("## FAQ")

        with gr.Accordion("1. Annual Energy Output (kWh)", open=False):
            gr.Markdown(
                "The total amount of electricity the system generates in one year.\n\n"
                "**Example**: A 5 kW system at 1,000 kWh/kW-year produces ~5,000 kWh/year.\n\n"
                "**Tip**: Check your system specifications or solar calculator output."
            )

        with gr.Accordion("2. Grid Emission Factor (kg CO₂/kWh)", open=False):
            gr.Markdown(
                "The amount of CO₂ emitted per kWh of electricity from the grid.\n\n"
                "**Example**: If your utility reports 0.5 kg CO₂/kWh, then 1 kWh offset avoids 0.5 kg CO₂.\n\n"
                "**Sources**: Local grid operator or national emission inventory."
            )

        with gr.Accordion("3. Embodied Energy (kWh)", open=False):
            gr.Markdown(
                "The total energy consumed to manufacture, transport, and install the solar system.\n\n"
                "**Example**: A 5 kW system may require ~15,000 kWh embodied energy.\n\n"
                "**Note**: Includes raw materials, fabrication, and logistics."
            )

        with gr.Accordion("4. Lifecycle Emissions (kg CO₂-eq)", open=False):
            gr.Markdown(
                "The total greenhouse gas emissions over the system's lifetime, expressed in CO₂ equivalents.\n\n"
                "**Example**: 10,000 kg CO₂-eq covers production, maintenance, and decommissioning.\n\n"
                "**Tip**: Use LCA studies or standard databases (e.g., ISO 14040)."
            )

        with gr.Accordion("5. System Lifetime (years)", open=False):
            gr.Markdown(
                "The expected operational lifespan of your PV system.\n\n"
                "**Typical Range**: 20-30 years.\n\n"
                "**Tip**: Manufacturer warranty often indicates reliable lifetime."
            )

        with gr.Accordion("6. How is CO₂ Avoided (tons/year) Calculated?", open=False):
            gr.Markdown(
                "Annual CO₂ emissions avoided by generating solar power instead of grid electricity.\n\n"
                "**Formula**:"
                "$$\\text{CO₂ Avoided} = \\frac{\\text{Annual Output} \\times \\text{Grid Emission}}{1000}$$\n\n"
                "**Example**: 5,000 kWh x 0.5 kg/kWh = 2,500 kg = 2.5 tons."
            )

        with gr.Accordion(
            "7. How is Energy Payback Time (years) Calculated?", open=False
        ):
            gr.Markdown(
                "Time required for the system to 'pay back' its embodied energy through clean energy generation.\n\n"
                "**Formula**:"
                "$$\\text{Payback Time} = \\frac{\\text{Embodied Energy}}{\\text{Annual Output}}$$\n"
                "**Example**: 15,000 kWh / 5,000 kWh/year = 3 years."
            )

        with gr.Accordion(
            "8. How is GHG Intensity (g CO₂-eq/kWh) Calculated?", open=False
        ):
            gr.Markdown(
                "Greenhouse gas emissions per kWh over the system's lifecycle.\n\n"
                "**Formula**:"
                "$$\\text{Emission Intensity} = \\frac{\\text{Lifecycle Emissions} \\times 1000}{\\text{Annual Output} \\times \\text{System Lifetime}}$$\n\n"
                "**Note**: If System Lifetime = 25 years, then for 10,000 kg CO₂-eq: (10,000x1000) / (5,000x25) = 80 g CO₂-eq/kWh."
            )


class EstimateTab:
    def __init__(self, parent):
        self.parent = parent

    def __call__(self):
        with gr.Tab("ROI"):
            ROI(self.parent)

        with gr.Tab("Payback"):
            Payback(self.parent)

        with gr.Tab("Solar System Sizing"):
            SolarSystemSizing(self.parent)

        with gr.Tab("Energy Yield"):
            EnergyYield(self.parent)

        with gr.Tab("Environmental Impact"):
            EnvironmentalImpact(self.parent)
