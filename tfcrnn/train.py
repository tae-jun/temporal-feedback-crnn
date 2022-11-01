import wandb
from tfcrnn.runners import SpeechCommandsRunner
from tfcrnn.config import Config


def main():
  config = Config()
  config.init_wandb()
  config.parse_cli()
  config.print()
  
  runner = SpeechCommandsRunner(config)
  
  print(runner.model)
  print(f'\n=> Num params: {sum([p.numel() for p in runner.model.parameters()]):,}')
  
  epoch = 0
  for stage in range(config.num_decay):
    print('-' * 80)
    print(f'Stage {stage}')
    print('-' * 80)
    
    for epoch in range(epoch + 1, config.num_max_epochs + 1):
      print(f'Epoch {epoch:2}')
      loss_train, scores_train = runner.train()
      loss_valid, scores_valid = runner.validate()
      loss_test, scores_test = runner.test()
      
      # Log the learning rate to watch lr decay.
      log = {
        'lr': runner.lr,
        'epoch': epoch,
        'loss_train': loss_train,
        'loss_valid': loss_valid,
        'loss_test': loss_test,
        'num_trained_samples': runner.total_trained_samples,
        **{f'{k}_train': v for k, v in scores_train.items()},
        **{f'{k}_valid': v for k, v in scores_valid.items()},
        **{f'{k}_test': v for k, v in scores_test.items()},
      }
      wandb.log(log, step=runner.total_trained_steps)
      
      is_best = runner.is_best(loss_valid)
      should_stop = runner.early_stop(loss_valid)
      if is_best:
        runner.save_checkpoint(**log)
        print(f'=> Checkpoint saved.')
      if should_stop:
        break
    
    # End of stage
    checkpoint = runner.load_checkpoint()  # back to the best weights
    print(f'=> The end of stage {stage}. Checkpoint loaded.')
  
  final_scores = {
    f'final_{k.replace("_test", "")}': checkpoint[k]
    for k in checkpoint.keys() if k.endswith('_test')
  }
  wandb.log(final_scores)
  print()
  for k, v in final_scores.items():
    print(f'=> {k:12}: {v:.4f}')
  print('=> Done.')


if __name__ == '__main__':
  main()
