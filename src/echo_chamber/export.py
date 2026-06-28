def export_dataframe(dataframe, output_path):
    """
    Save a DataFrame to CSV.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(output_path, index=False)


def export_project_results(clusters, polarization_report, cluster_output, polarization_output):
    """
    Export the main project result tables.
    """

    export_dataframe(clusters, cluster_output)
    export_dataframe(polarization_report, polarization_output)