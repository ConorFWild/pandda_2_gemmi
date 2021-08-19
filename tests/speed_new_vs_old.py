import fire

def parse_old_pandda_log_for_speed(new_pandda_path):

    log_path = new_pandda_path / "pandda_log.json"

def speed_from_old_pandda(old_pandda_path):
    log_path = next((old_pandda_path / "logs").glob("*.log"))

    return parse_old_pandda_log_for_speed(log_path)

def cacluate_speed_old_vs_new(pandda_dir, old_pandda_dir):

    new_speed = speed_from_new_pandda()

    old_speed = speed_from_old_pandda()

    return (old, new_speed)

if __name__ == "__main__":
    fire.Fire(cacluate_speed_old_vs_new)