import importlib.resources as imp_res
import os
from typing import List, Tuple

from bs4 import BeautifulSoup

from zppy_interfaces.global_time_series.coupled_global_utils import Metric, Variable
from zppy_interfaces.global_time_series.utils import Parameters
from zppy_interfaces.multi_utils.logger import _setup_custom_logger
from zppy_interfaces.multi_utils.viewer import OutputViewer

logger = _setup_custom_logger(__name__)


class VariableGroup(object):
    def __init__(self, name: str, variables: List[Variable]):
        self.group_name = name
        self.variables = variables


def get_variable_groups(variables: List[Variable]) -> List[VariableGroup]:
    group_names: List[str] = []
    groups: List[VariableGroup] = []
    for v in variables:
        g: str = v.group
        if g not in group_names:
            # A new group!
            group_names.append(g)
            groups.append(VariableGroup(g, [v]))
        else:
            # Add a new variable to this existing group
            for group in groups:
                if g == group.group_name:
                    group.variables.append(v)
    return groups


def create_viewer(parameters: Parameters, vars: List[Variable], component: str) -> str:
    logger.info(f"Creating viewer for {component}")
    index_name = f"zppy global time-series plot: {parameters.experiment_name} {component} component ({parameters.year1}-{parameters.year2})"
    viewer = OutputViewer(path=parameters.results_dir, index_name=index_name)
    viewer.add_page(f"table_{component}", parameters.regions)
    groups: List[VariableGroup] = get_variable_groups(vars)
    for group in groups:
        logger.info(f"Adding group {group.group_name}")
        # Only groups that have at least one variable will be returned by `get_variable_groups`
        # So, we know this group will be non-empty and should therefore be added to the viewer.
        viewer.add_group(group.group_name)
        for var in group.variables:
            plot_name: str = var.variable_name
            row_title: str
            if var.metric == Metric.AVERAGE:
                metric_name = "AVERAGE"
            elif var.metric == Metric.TOTAL:
                metric_name = "TOTAL"
            else:
                # This shouldn't be possible
                raise ValueError(f"Invalid Enum option for metric={var.metric}")
            if var.long_name != "":
                row_title = f"{plot_name}: {var.long_name}, metric={metric_name}"
            else:
                row_title = f"{plot_name}, metric={metric_name}"
            viewer.add_row(row_title)
            for rgn in parameters.regions:
                viewer.add_col(
                    f"{parameters.figstr}_{rgn}_{component}_{plot_name}.png",
                    is_file=True,
                    title=f"{rgn}_{component}_{plot_name}",
                )

    url = viewer.generate_page()
    viewer.generate_viewer()
    # Example links:
    # Viewer is expecting the actual images to be in the directory above `table`.
    # table/index.html links to previews with: ../v3.LR.historical_0051_glb_lnd_FSH.png
    # Viewer is expecting individual image html pages to be under both group and var subdirectories.
    # table/energy-flux/fsh-sensible-heat/glb_lnd_fsh.html links to: ../../../v3.LR.historical_0051_glb_lnd_FSH.png
    return url


# Copied from E3SM Diags and modified
def create_viewer_index(
    root_dir: str, title_and_url_list: List[Tuple[str, str]]
) -> str:
    """
    Creates the index page in root_dir which
    joins the individual viewers.
    Each tuple is on its own row.
    """

    logger.info("Creating viewer index")

    def insert_data_in_row(row_obj, name: str, url: str):
        """
        Given a row object, insert the name and url.
        """
        td = soup.new_tag("td")
        a = soup.new_tag("a")
        a["href"] = url
        a.string = name
        td.append(a)
        row_obj.append(td)

    path: str = str(
        imp_res.files("zppy_interfaces.global_time_series") / "index_template.html"
    )
    output: str = os.path.join(root_dir, "index.html")

    soup = BeautifulSoup(open(path), "lxml")

    # If no one changes it, the template only has
    # one element in the find command below.
    table = soup.find_all("table", {"class": "table"})[0]

    # Adding the title.
    tr = soup.new_tag("tr")
    th = soup.new_tag("th")
    th.string = "Output Sets"
    tr.append(th)

    # Adding each of the rows.
    for row in title_and_url_list:
        tr = soup.new_tag("tr")

        if isinstance(row, list):
            for elt in row:
                name, url = elt
                insert_data_in_row(tr, name, url)
        else:
            name, url = row
            insert_data_in_row(tr, name, url)

        table.append(tr)

    html = soup.prettify("utf-8")

    with open(output, "wb") as f:
        f.write(html)

    return output
