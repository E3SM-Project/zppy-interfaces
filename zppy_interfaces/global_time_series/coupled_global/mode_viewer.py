import importlib.resources as imp_res
import os
from typing import List, Tuple

from bs4 import BeautifulSoup

from zppy_interfaces.global_time_series.coupled_global.mix_viewer_component import (
    create_viewer_for_component,
    get_vars_component,
)
from zppy_interfaces.global_time_series.coupled_global.mix_viewer_original import (
    create_viewer_for_original,
)
from zppy_interfaces.global_time_series.coupled_global.utils import (
    RequestedVariables,
    Variable,
)
from zppy_interfaces.global_time_series.utils import Parameters
from zppy_interfaces.multi_utils.logger import _setup_child_logger

logger = _setup_child_logger(__name__)

# This file is for making Viewers
# Hence, "mode_viewer"

# Used by driver.run_coupled_global ###########################################


def produce_viewer(parameters: Parameters, requested_variables: RequestedVariables):
    title_and_url_list: List[Tuple[str, str]] = []
    for component in [
        "original",
        "atm",
        "ice",
        "lnd",
        "ocn",
    ]:
        if component == "original":
            if parameters.plots_original:
                url = create_viewer_for_original(parameters)
            else:
                continue
        else:
            vars_list: List[Variable] = get_vars_component(
                requested_variables, component
            )
            if vars_list:
                url = create_viewer_for_component(parameters, vars_list, component)
            else:
                continue
        logger.info(f"Viewer URL for {component}: {url}")
        title_and_url_list.append((component, url))
    index_url: str = _create_viewer_index(parameters.results_dir, title_and_url_list)
    logger.info(f"Viewer index URL: {index_url}")


# Copied from E3SM Diags and modified
def _create_viewer_index(
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
