
import os
import sys
import datetime
import logging
import argparse
import provider_data
import vsa
from pathlib import Path
import time
from metrics import *
from torch.utils.tensorboard import SummaryWriter
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
TOTAL_BAR_LENGTH = 65.

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='3', help='specify gpu device')
    parser.add_argument("--eyetracking_log_path", type=str, default='./tensorboard_log/', help="path to eyetracking_log")
    parser.add_argument("--log_name", type=str, default='/ini30/v14_new', help="path to eyetracking_log")
    parser.add_argument('--train_h5_path', type=str, default='./data/ini30v2/train.h5', help='train_data')
    parser.add_argument('--test_h5_path', type=str, default='./data/ini30v2/test.h5', help='test_data')
    parser.add_argument('--save_path', type=str, default='./checkpoint', help='model_path')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size in training')
    parser.add_argument('--sensor_width', type=int, default=512, help='sensor width')
    parser.add_argument('--sensor_height', type=int, default=512, help='sensor height')
    parser.add_argument('--spatial_factor', type=float, default=0.125, help='spatial factor')
    parser.add_argument('--num_category', default=2, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--pixel_tolerances', default=[3, 5, 10, 15], type=int,  help='pixel_tolerances')
    parser.add_argument('--epoch', default=450, type=int, help='number of epoch in training')
    parser.add_argument('--loss', default='weighted_mse', type=str, help='loss')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer for training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--isELL', type=bool, default=True, help='ELL')                    #是否使用椭圆全部信息
    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def validate(net, val_loader, criterion,A):
    net.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_a = 0.0
    total_b = 0.0
    total_angle = 0.0
    total_p_corr_all = {f'p{p}_all': 0 for p in args.pixel_tolerances}
    total_p_error_all = {f'error_all': 0}
    total_samples_all, total_sample_p_error_all = 0, 0
    with torch.no_grad():
        for data, label in val_loader:
            data = data.permute(0, 2, 1)
            label = label.cuda()
            target_VSA = vsa.Encode_VSA(label,A,isELL=args.isELL)
            target_vector = torch.cat([target_VSA.real, target_VSA.imag], dim=1)
            predict_vector = net(data.cuda())
            loss = criterion(predict_vector, target_vector)
            total_loss += loss.item()
            real_part_p, imag_part_p = torch.chunk(predict_vector, chunks=2, dim=-1)
            predict_VSA  = torch.complex(real_part_p, imag_part_p)
            outputs = vsa.Decode_VSA(predict_VSA,A,isELL=args.isELL)
            p_corr, batch_size = p_acc(label[:, :2], outputs[:, :2],
                                       width_scale=args.sensor_width * args.spatial_factor,
                                       height_scale=args.sensor_height * args.spatial_factor,
                                       pixel_tolerances=args.pixel_tolerances)
            total_p_corr_all = {f'p{k}_all': (total_p_corr_all[f'p{k}_all'] + p_corr[f'p{k}']).item() for k in
                                args.pixel_tolerances}
            total_samples_all += batch_size
            p_error_total, bs_times_seqlen = px_euclidean_dist(label[:, :2], outputs[:, :2],
                                                               width_scale=args.sensor_width * args.spatial_factor,
                                                               height_scale=args.sensor_height * args.spatial_factor)
            total_p_error_all = {f'error_all': (total_p_error_all[f'error_all'] + p_error_total).item()}
            total_sample_p_error_all += bs_times_seqlen
            if args.isELL == True:
                aerror,berror, _ = px_euclidean_ab(label[:, 2:4], outputs[:, 2:4],
                                                                width_scale=args.sensor_width * args.spatial_factor,
                                                                height_scale=args.sensor_height * args.spatial_factor)
                angleerror, _ = px_euclidean_angle(label[:, -1], outputs[:, -1])
                total_a += aerror.item()
                total_b += berror.item()
                total_angle += angleerror.item()
                iou = compute_ellipse_iou(outputs,label,args.sensor_width,args.sensor_height)
                total_iou += iou.item()
    if args.isELL == True:
        metrics = {'val_p_acc_all': {f'val_p{k}_acc_all': (total_p_corr_all[f'p{k}_all'] / total_samples_all) for k in
                                    args.pixel_tolerances},
                'val_p_error_all': {f'val_p_error_all': (total_p_error_all[f'error_all'] / total_sample_p_error_all)},
                'loss':total_loss / len(val_loader),
                'IOU':total_iou / total_samples_all,
                'total_a':total_a / total_samples_all,
                'total_b':total_b / total_samples_all,
                'total_angle':total_angle / total_samples_all
                }
    else:
        metrics = {'val_p_acc_all': {f'val_p{k}_acc_all': (total_p_corr_all[f'p{k}_all'] / total_samples_all) for k in
                                    args.pixel_tolerances},
                'val_p_error_all': {f'val_p_error_all': (total_p_error_all[f'error_all'] / total_sample_p_error_all)},
                'loss':total_loss / len(val_loader),
                }
    return total_loss / len(val_loader), metrics

def train(net, trainloader, optimizer, criterion,device,A):
    net.train()
    total_loss = 0.0
    total_iou = 0.0
    total_a = 0.0
    total_b = 0.0
    total_angle = 0.0
    total_p_corr_all = {f'p{p}_all': 0 for p in args.pixel_tolerances}
    total_p_error_all = {f'error_all': 0}
    total_samples_all, total_sample_p_error_all = 0, 0
    for batch_idx, (data, label) in enumerate(trainloader):
        data = data.permute(0, 2, 1)
        data, label = data.cuda(), label.cuda().squeeze()
        optimizer.zero_grad()
        target_VSA = vsa.Encode_VSA(label,A,isELL=args.isELL)
        target_vector = torch.cat([target_VSA.real, target_VSA.imag], dim=1)
        predict_vector = net(data)
        loss = criterion(predict_vector, target_vector)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        real_part_p, imag_part_p = torch.chunk(predict_vector, chunks=2, dim=-1)
        predict_VSA  = torch.complex(real_part_p, imag_part_p)
        outputs = vsa.Decode_VSA(predict_VSA,A,isELL=args.isELL)
        p_corr, batch_size = p_acc(label[:, :2], outputs[:, :2],
                                   width_scale=args.sensor_width * args.spatial_factor,
                                   height_scale=args.sensor_height * args.spatial_factor,
                                   pixel_tolerances=args.pixel_tolerances)
        total_p_corr_all = {f'p{k}_all': (total_p_corr_all[f'p{k}_all'] + p_corr[f'p{k}']).item() for k in
                            args.pixel_tolerances}
        total_samples_all += batch_size
        p_error_total, bs_times_seqlen = px_euclidean_dist(label[:, :2], outputs[:, :2],
                                                           width_scale=args.sensor_width * args.spatial_factor,
                                                           height_scale=args.sensor_height * args.spatial_factor)
        total_p_error_all = {f'error_all': (total_p_error_all[f'error_all'] + p_error_total).item()}
        total_sample_p_error_all += bs_times_seqlen
        if args.isELL == True:
            aerror,berror, _ = px_euclidean_ab(label[:, 2:4], outputs[:, 2:4],
                                                                width_scale=args.sensor_width * args.spatial_factor,
                                                                height_scale=args.sensor_height * args.spatial_factor)
            angleerror, _ = px_euclidean_angle(label[:, -1], outputs[:, -1])
            total_a += aerror.item()
            total_b += berror.item()
            total_angle += angleerror.item()
            iou = compute_ellipse_iou(outputs,label,args.sensor_width,args.sensor_height)
            total_iou += iou.item()
    if args.isELL == True:
        metrics = {'tr_p_acc_all': {f'tr_p{k}_acc_all': (total_p_corr_all[f'p{k}_all'] / total_samples_all) for k in
                                    args.pixel_tolerances},
                'tr_p_error_all': {f'tr_p_error_all': (total_p_error_all[f'error_all'] / total_sample_p_error_all)},
                'loss':total_loss / len(trainloader),
                'IOU':total_iou / total_samples_all,
                'total_a':total_a / total_samples_all,
                'total_b':total_b / total_samples_all,
                'total_angle':total_angle / total_samples_all,
                }
    else:
        metrics = {'tr_p_acc_all': {f'tr_p{k}_acc_all': (total_p_corr_all[f'p{k}_all'] / total_samples_all) for k in
                                    args.pixel_tolerances},
                'tr_p_error_all': {f'tr_p_error_all': (total_p_error_all[f'error_all'] / total_sample_p_error_all)},
                'loss':total_loss / len(trainloader),
                }
    return net, total_loss / len(trainloader), metrics

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('eye_tracking')
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eventmamba.txt' % (log_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    current_data_train, current_label_train = provider_data.load_h5_and_resample_INI30(args.train_h5_path, sample_size=args.num_point)
    current_data_train = np.array(current_data_train).astype(np.float32)
    current_label_train = np.array(current_label_train).astype(np.float32)
    current_data_train = torch.from_numpy(current_data_train)
    current_label_train = torch.from_numpy(current_label_train)
    dataset = torch.utils.data.TensorDataset(current_data_train, current_label_train)
    trainDataLoader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
      

    current_data_test, current_label_test = provider_data.load_h5_and_resample_INI30(args.test_h5_path, sample_size=args.num_point)
    current_data_test = np.array(current_data_test).astype(np.float32)
    current_label_test = np.array(current_label_test).astype(np.float32)
    current_data_test = torch.from_numpy(current_data_test)
    current_label_test = torch.from_numpy(current_label_test)
    dataset_test = torch.utils.data.TensorDataset(current_data_test, current_label_test)
    testDataLoader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    '''MODEL LOADING'''
    from models.eventmamba_v3 import EventMamba
    classifier = EventMamba(num_classes=args.num_category)
    classifier.apply(inplace_relu)
    
    '''VSA LOADING'''  
    VSA_dim = 512
    A = vsa.generate_random_matrix_A(2,VSA_dim)
    if not args.use_cpu:
        A = A.cuda()
    if not os.path.exists(args.save_path+args.log_name):
        os.makedirs(args.save_path+args.log_name)
    torch.save(A, os.path.join(args.save_path+args.log_name, "matrix_A.pt"))

    if not args.use_cpu:
        classifier = classifier.cuda()

    if args.isELL == True:
        criterion = torch.nn.SmoothL1Loss()
    else:
        criterion = CosineSimilarityLoss()

    
    try:
        checkpoint = torch.load('./last_checkpoint.pth')
        start_epoch = 0
        classifier.load_state_dict(checkpoint)
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0


    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 300])

    '''TRANING'''
    logger.info('Start training...')

    best_p3 = 0
    best_iou = 0
    writer = SummaryWriter(args.eyetracking_log_path + args.log_name)
    for epoch in range(start_epoch, args.epoch):
        classifier = classifier.train()
        net, train_loss, metrics = train(classifier, trainDataLoader, optimizer, criterion,'cuda',A)
        writer.add_scalar('Train/Pixel_Accuracy/tr_p3_acc_all', metrics['tr_p_acc_all']['tr_p3_acc_all'], epoch)
        writer.add_scalar('Train/Pixel_Accuracy/tr_p5_acc_all', metrics['tr_p_acc_all']['tr_p5_acc_all'], epoch)
        writer.add_scalar('Train/Pixel_Accuracy/tr_p10_acc_all', metrics['tr_p_acc_all']['tr_p10_acc_all'], epoch)
        writer.add_scalar('Train/Pixel_Accuracy/tr_p15_acc_all', metrics['tr_p_acc_all']['tr_p15_acc_all'], epoch)
        writer.add_scalar('Train/Pixel_Error/tr_p_error_all', metrics['tr_p_error_all']['tr_p_error_all'], epoch)
        writer.add_scalar('Train/loss',metrics['loss'], epoch)
        if args.isELL == True:
            writer.add_scalar('Train/IOU',metrics['IOU'], epoch)
            writer.add_scalar('Train/a_error',metrics['total_a'], epoch)
            writer.add_scalar('Train/b_error',metrics['total_b'], epoch)
            writer.add_scalar('Train/angle_error',metrics['total_angle'], epoch)
        val_loss, val_metrics = validate(net, testDataLoader, criterion,A)
        if val_metrics['val_p_acc_all']['val_p3_acc_all'] > best_p3:
            best_p3 = val_metrics['val_p_acc_all']['val_p3_acc_all']
            print("best_p3",val_metrics['val_p_acc_all']['val_p3_acc_all'], "best_p5",val_metrics['val_p_acc_all']['val_p5_acc_all'], "best_p10",val_metrics['val_p_acc_all']['val_p10_acc_all'], "best_p15",val_metrics['val_p_acc_all']['val_p15_acc_all'])
            if not os.path.exists(args.save_path+args.log_name):
                os.makedirs(args.save_path+args.log_name)
            torch.save(net.state_dict(), os.path.join(args.save_path+args.log_name, "P3best_checkpoint.pth"))
        if args.isELL == True:
            if val_metrics['IOU'] > best_iou:
                best_iou = val_metrics['IOU']
                print("best_iou",val_metrics['IOU'])
                if not os.path.exists(args.save_path+args.log_name):
                    os.makedirs(args.save_path+args.log_name)
                torch.save(net.state_dict(), os.path.join(args.save_path+args.log_name, "IOUbest_checkpoint.pth"))
        torch.save(net.state_dict(), os.path.join(args.save_path+args.log_name, "last_checkpoint.pth"))
        print(f"[Validation] at Epoch {epoch + 1}/{args.epoch}: Val Loss: {val_loss:.4f}")
        writer.add_scalar('Val/Pixel_Accuracy/val_p3_acc_all', val_metrics['val_p_acc_all']['val_p3_acc_all'], epoch)
        writer.add_scalar('Val/Pixel_Accuracy/val_p5_acc_all', val_metrics['val_p_acc_all']['val_p5_acc_all'], epoch)
        writer.add_scalar('Val/Pixel_Accuracy/val_p10_acc_all', val_metrics['val_p_acc_all']['val_p10_acc_all'], epoch)
        writer.add_scalar('Val/Pixel_Accuracy/val_p15_acc_all', val_metrics['val_p_acc_all']['val_p15_acc_all'], epoch)
        writer.add_scalar('Val/Pixel_Error/val_p_error_all', val_metrics['val_p_error_all']['val_p_error_all'], epoch)
        writer.add_scalar('Val/loss',val_metrics['loss'], epoch)
        if args.isELL == True:
            writer.add_scalar('Val/IOU',val_metrics['IOU'], epoch)
            writer.add_scalar('Val/a_error',val_metrics['total_a'], epoch)
            writer.add_scalar('Val/b_error',val_metrics['total_b'], epoch)
            writer.add_scalar('Val/angle_error',val_metrics['total_angle'], epoch)
            print('val_a_error=',val_metrics['total_a'])
            print('val_b_error=',val_metrics['total_b'])
            print('val_angle_error=',val_metrics['total_angle'])
            print('val-IOU=',val_metrics['IOU'])
        print('val_acc=', val_metrics['val_p_acc_all'])
        print('val_error=', val_metrics['val_p_error_all'])

        print(f"Epoch {epoch+1}/{args.epoch}: Train Loss: {train_loss:.4f}")
        scheduler.step()
    writer.close()
    logger.info('End of training...')
if __name__ == '__main__':
    args = parse_args()
    main(args)