import bokeh.plotting
import bokeh.models
import bokeh.io
import bokeh.layouts
import bokeh.palettes

import itertools


class Columns:
    def __init__(
        self, arrangement, x_labels = None, y_labels = None, *, palette=None, height=None, width=None
    ):
        """arrangement looks like [{'x': x_key1, 'y', [y_key1, y_key2]}, {'x': x_key2, 'y', [y_key3, y_key4, y_key5]}]"""

        if palette is None:
            palette = bokeh.palettes.Category10[10]

        self.palette = palette
        self.colors = itertools.cycle(self.palette)

        self.figures = {}

        self.y_labels = y_labels
        self.x_labels = x_labels
        self.x_ranges = {}

        self.arrangement = arrangement

        self.columns = []

        self.legend = bokeh.models.Legend(orientation="vertical")

        tool_hover = bokeh.models.HoverTool(
            tooltips=[("x", "$snap_x"), ("y", "$snap_y")]
        )
        
        self.variables = set()

        for keys_column in self.arrangement:
            x_key = keys_column["x"]
            y_keys = keys_column["y"]

            self.variables.add(x_key)
            self.variables.update(y_keys)
            
            row_count = len(y_keys)
            
            for i, y_key in enumerate(y_keys):
                key_figure = (x_key, y_key)

                figure = self._create_figure(key_figure)

                # TODO: Move the stuff below into _create_figure()

                if i == 0:
                    self.x_ranges[x_key] = figure.x_range
                else:
                    figure.x_range = self.x_ranges[x_key]

                if i != row_count - 1:
                    figure.xaxis.visible = False
                elif x_labels is not None and x_key in x_labels and x_labels[x_key] is not None:
                    figure.xaxis.axis_label = x_labels[x_key]

                figure.add_tools(tool_hover)

                if y_labels is not None and y_key in y_labels and self.y_labels[y_key] is not None:
                    figure.yaxis.axis_label = self.y_labels[y_key]

                self.figures[key_figure] = figure

            children = [[self.figures[(x_key, y_key)]] for y_key in y_keys]

            gridplot = bokeh.layouts.gridplot(
                children=children,
                # height = height,
                sizing_mode="stretch_both",
            )

            self.columns.append(gridplot)

        # plot_legend = bokeh.plotting.figure(width = 300, sizing_mode = 'stretch_height')
        # plot_legend.add_layout(self.legend)

        # self.layout = bokeh.layouts.row(self.columns + [plot_legend], sizing_mode = 'stretch_width')

        if height is None and width is None:
            sizing_mode = "stretch_both"
        elif height is None:
            sizing_mode = "stretch_height"
        elif width is None:
            sizing_mode = "stretch_width"
        else:
            sizing_mode = "fixed"

        self.layout = bokeh.layouts.row(
            self.columns, sizing_mode=sizing_mode, height=height, width=width
        )

        self._format_figures()

        # self.layout.add_layout(bokeh.models.Legend())
        # self.layout.legend.click_policy="hide"

    def add_trace(
        self,
        column_data_source=None,
        *,
        color_index=None,
        label=None,
        style={},
        **kwargs,
    ):
        style = style.copy()

        if "line_color" not in style and "fill_color" not in style:
            if color_index is None:
                color_trace = next(self.colors)
            else:
                color_trace = self.palette[color_index]

        data_keymap = {}
        data_direct = {}

        for key in self.variables:
            if column_data_source is not None:
                if key in kwargs:
                    data_keymap[key] = kwargs.pop(key)
                elif key in column_data_source.column_names:
                    data_keymap[key] = key
                else:
                    raise (
                        KeyError(f"No column '{key}' given or matched from constructor")
                    )
            else:
                if key in kwargs:
                    data_keymap[key] = key
                    data_direct[key] = kwargs.pop(key)
                else:
                    raise (KeyError(f"No column '{key}' data given"))

        # Create a data source if we were only passed individual columns
        if column_data_source is None:
            column_data_source = bokeh.models.ColumnDataSource(data_direct)

        glyphs = []
        i = 0
        for key_figure in self.figures.keys():
            key_data = (data_keymap[key_figure[0]], data_keymap[key_figure[1]])

            glyphs += self._plot_trace(
                column_data_source, key_data, key_figure, style, color_trace
            )

            if i == 0:
                # remove keywords that we don't want to apply to the lower plots
                for keyword in ["legend_label", "legend_group"]:
                    style.pop(keyword, None)
            i += 1

        # Do any chart specific updating that we need to do
        self._add_trace(column_data_source, color_index, label, style)
        # self._add_legend_entry(label, glyphs)

    def _plot_trace(self, column_data_source, key_data, key_figure, style, color_trace):
        line = self.figures[key_figure].line(
            source=column_data_source,
            x=key_data[0],
            y=key_data[1],
            color=color_trace,
            **style,
        )
        return [line]

    def _add_trace(self, column_data_source, color_index, label, style):
        pass

    def _add_legend_entry(self, label, glyphs):
        if label is not None:
            legend_item = bokeh.models.LegendItem(label=label, renderers=glyphs)
            self.legend.items.append(legend_item)

    def _format_figures(self):
        pass

    def _create_figure(self, key_figure):
        return bokeh.plotting.figure()
    
    def show(self):
        bokeh.io.show(self.layout)


class Stacked(Columns):
    def __init__(self, x_label, y_labels, **kwargs):
        arrangement = [{"x": list(x_label.keys())[0], "y": list(y_labels.keys())}]

        super().__init__(arrangement, x_label, y_labels, **kwargs)


class Chrono(Stacked):
    def _plot_trace(self, data, key_data, key_figure, style, color_trace):
        line = self.figures[key_figure].line(
            source=data, x=key_data[0], y=key_data[1], color=color_trace, **style
        )
        return [line]


class StackedSlider(Stacked):
    def __init__(self, *, x_label, y_labels, slider_type="index", **kwargs):
        # Create a slider to step through simulation timesteps
        self.slider = bokeh.models.Slider(
            start=0, end=0.1, value=0, sizing_mode="stretch_width"
        )

        if slider_type == "index":
            filter_code = """
                             const index_max = source.get_length() -1;
                             const index = slider.value;

                             if  (index >= index_max)
                             { 
                                 return [index_max];
                             }
                             else
                             {
                                 return [index];
                             }
                          """
        elif slider_type == "time":
            filter_code == time

        self.filter = bokeh.models.CustomJSFilter(
            args={"slider": self.slider}, code=filter_code
        )

        # Init the base class so it constructs the layout
        super().__init__(x_label=x_label, y_labels=y_labels, **kwargs)

        # Place the base layout inside a column with a slider to step through data
        self.layout = bokeh.layouts.column(
            self.layout, self.slider, sizing_mode="stretch_width"
        )

    def _plot_trace(self, column_data_source, key_data, key_figure, style, color_trace):
        # print(f"key_data = {key_data}, key_figure = {key_figure}")

        view = bokeh.models.CDSView(filter=self.filter)

        self.slider.end = max(
            self.slider.end, len(column_data_source.data[key_data[0]]) - 1
        )

        multi_line = self.figures[key_figure].multi_line(
            source=column_data_source,
            xs=key_data[0],
            ys=key_data[1],
            view=view,
            color=color_trace,
            **style,
        )

        self.slider.js_on_change(
            "value",
            bokeh.models.CustomJS(
                args={"source": column_data_source}, code="""source.change.emit()"""
            ),
        )
        return [multi_line]


class CrossSection(Columns):
    def __init__(
        self,
        arrangement,
        x_labels,
        y_labels,
        slider_type="index",
        slider_value=None,
        **kwargs,
    ):
        if len(arrangement) != 2:
            raise ValueError("arrangement must be length 2 for this plot type")

        # Construct a filter to cross section the data to only what the slider shows

        if slider_type == "index":
            # Create a slider to step through simulation timesteps

            if slider_value is None:
                slider_value = 0

            self.slider = bokeh.models.Slider(
                start=0,
                end=0.1,
                step=0.1,
                value=slider_value,
                sizing_mode="stretch_width",
            )

            filter_code = """
                             const index_max = source.get_length() -1;
                             const time = slider.value;

                            for (let i = 0; i < index_max; i++)
                                 if  (source.data.t[i] >= time)
                                 { 
                                     return [i];
                                 }

                            return [index_max]
                          """

            # Construct a span to show our current cross section
            self.annotation_crossect = bokeh.models.Span(dimension="height", location=0)

            self.slider.js_link("value", self.annotation_crossect, "location")

        elif slider_type == "range":
            if slider_value is None:
                slider_value = (0, 0.1)

            self.slider = bokeh.models.RangeSlider(
                start=0,
                end=0.1,
                step=0.1,
                value=slider_value,
                sizing_mode="stretch_width",
            )
            filter_code = """
                             const index_max = source.get_length() -1;
                             const times = slider.value;
                             let indices = []
                             
                            for (let i = 0; i < index_max; i++)
                                 if  (source.data.t[i] >= times[0] && source.data.t[i] <= times[1])
                                 { 
                                     indices.push(i);
                                 }

                            return indices
                          """

            self.annotation_crossect = bokeh.models.BoxAnnotation(
                left=0, right=0, fill_alpha=0.2, fill_color="black", line_color="black"
            )

            self.slider.js_link(
                "value", self.annotation_crossect, "left", attr_selector=0
            )
            self.slider.js_link(
                "value", self.annotation_crossect, "right", attr_selector=1
            )

        self.filter = bokeh.models.CustomJSFilter(
            args={"slider": self.slider}, code=filter_code
        )

        # self.tool_crossect = bokeh.models.TapTool(behavior = 'inspect',
        #                                           callback = bokeh.models.CustomJS(args = {'slider': self.slider},
        #                                                                            code = """slider.value = cb_data.geometries.x""")
        #                                          )

        # Save the first x axis key to use as a cross section
        self.key_section = arrangement[0]["x"]

        # Init the base class
        super().__init__(arrangement, x_labels, y_labels, **kwargs)

        # Place the base layout inside a column with a slider to step through data
        self.layout = bokeh.layouts.column(
            self.layout, self.slider, sizing_mode="stretch_width"
        )

    def _create_figure(self, key_figure):
        figure = bokeh.plotting.figure()

        # If we're in the sectioning column, add a vertical line that corresponds to the slider and a tool to move it
        if key_figure[0] == self.key_section:
            figure.add_layout(self.annotation_crossect)
            # figure.add_tools(self.tool_crossect)

        return figure

    def _add_trace(self, column_data_source, color_index, label, style):
        # Update the slider to end at the longest simulation.
        self.slider.end = max(
            self.slider.end, column_data_source.data[self.key_section][-1]
        )

    def _plot_trace(self, column_data_source, key_data, key_figure, style, color_trace):
        view = bokeh.models.CDSView(filter=self.filter)

        # Check which column we are in to plot the right type:

        if key_figure[0] == self.key_section:
            glyph = self.figures[key_figure].line(
                source=column_data_source,
                x=key_data[0],
                y=key_data[1],
                color=color_trace,
                **style,
            )
        else:
            glyph = self.figures[key_figure].multi_line(
                source=column_data_source,
                xs=key_data[0],
                ys=key_data[1],
                view=view,
                color=color_trace,
                **style,
            )

            self.slider.js_on_change(
                "value",
                bokeh.models.CustomJS(
                    args={"source": column_data_source}, code="""source.change.emit()"""
                ),
            )

        return [glyph]
