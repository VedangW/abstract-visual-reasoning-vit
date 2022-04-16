class TrainerAndEvaluator:

    def __init__(self, model):
        self.model = model

    def train(self, dataloader, epoch, args, save_file):
        self.model.train()

        loss_all, acc_all, counter = 0., 0., 0
        
        for batch_idx, (image, target, meta_target, meta_structure, embedding, indicator) in enumerate(dataloader):
            counter += 1

            if args.cuda:
                image = image.cuda()
                target = target.cuda()
                meta_target = meta_target.cuda()
                meta_structure = meta_structure.cuda()
                embedding = embedding.cuda()
                indicator = indicator.cuda()

            loss, acc = self.model.train_(image, target, meta_target, meta_structure, embedding, indicator)
            save_str = 'Train: Epoch:{}, Batch:{}, Loss:{:.6f}, Acc:{:.4f}'.format(epoch, batch_idx, loss, acc)

            if counter % 20 == 0:
                print(save_str)

            with open(save_file, 'a') as f:
                f.write(save_str + "\n")

            loss_all += loss
            acc_all += acc
            
        if counter > 0:
            save_str = "Train_: Avg Training Loss: {:.6f}, Avg Training Acc: {:.6f}".format(
                loss_all/float(counter),
                (acc_all/float(counter))
            )

            print(save_str)
            with open(save_file, 'a') as f:
                f.write(save_str + "\n")
                
        return loss_all/float(counter), acc_all/float(counter)


    def validate(self, dataloader, args, save_file):
        self.model.eval()

        loss_all, acc_all, counter = 0., 0., 0

        for _, (image, target, meta_target, meta_structure, embedding, indicator) in enumerate(dataloader):
            counter += 1

            if args.cuda:
                image = image.cuda()
                target = target.cuda()
                meta_target = meta_target.cuda()
                meta_structure = meta_structure.cuda()
                embedding = embedding.cuda()
                indicator = indicator.cuda()

            loss, acc = self.model.validate_(image, target, meta_target, meta_structure, embedding, indicator)
            loss_all += loss
            acc_all += acc

        if counter > 0:
            save_str = "Val_: Total Validation Loss: {:.6f}, Acc: {:.4f}".format((loss_all/float(counter)), (acc_all/float(counter)))
            print(save_str)

            with open(save_file, 'a') as f:
                f.write(save_str + "\n")

        return loss_all/float(counter), acc_all/float(counter)


    def test(self, dataloader, args, save_file):
        self.model.eval()

        acc_all, counter = 0., 0

        for _, (image, target, meta_target, meta_structure, embedding, indicator) in enumerate(dataloader):
            counter += 1

            if args.cuda:
                image = image.cuda()
                target = target.cuda()
                meta_target = meta_target.cuda()
                meta_structure = meta_structure.cuda()
                embedding = embedding.cuda()
                indicator = indicator.cuda()

            acc = self.model.test_(image, target, meta_target, meta_structure, embedding, indicator)
            acc_all += acc

        if counter > 0:
            save_str = "Test_: Total Testing Acc: {:.4f}".format((acc_all / float(counter)))
            print(save_str)

            with open(save_file, 'a') as f:
                f.write(save_str + "\n")

        return acc_all/float(counter)