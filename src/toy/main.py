import pprint
import sys

import hydra
from omegaconf import OmegaConf

import conf.conf
import toy.steering_experiment
import toy.utils

log = toy.utils.get_logger_by_filename(__file__)
hydra_init = toy.utils.HYDRA_INIT


def run(cfg):
    with conf.conf.Session() as db:
        cfg: conf.conf.Conf = conf.conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        db.commit()
        cfg.run_dir.mkdir(parents=True, exist_ok=True)

        cmd_args = sys.argv
        if cmd_args[-1].startswith('hydra.run.dir='):
            cmd_args = cmd_args[:-1]
        log.info('Command: python %s', ' '.join(cmd_args))
        log.info(pprint.pformat(cfg))
        log.info('Output directory: %s', cfg.run_dir)

    samples_df = toy.steering_experiment.run_and_make_samples_df(cfg)
    samples_csv_path = cfg.run_dir / 'samples.csv'
    samples_df.write_csv(samples_csv_path)
    log.info('Saved samples CSV: %s', samples_csv_path)


@hydra.main(**hydra_init)
def main(cfg):
    run(cfg)


if __name__ == '__main__':
    last_override, run_dir = conf.conf.get_run_dir(hydra_init=hydra_init)
    conf.conf.set_run_dir(last_override, run_dir)
    main()
