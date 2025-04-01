import torch
import torch.nn as nn

class FP(nn.Module):
    def __init__(self, viewNum, chanNum, batchSize):
        super(FP, self).__init__()
        self.viewNum = viewNum
        self.chanNum = chanNum
        self.batchSize = batchSize
    def forward(self, x):
        '''
            x: image 
            x is a tensor (batchSize*netChanNum*imgSize*imgSize)
        '''
        sino = torch.from_numpy( np.zeros((self.batchSize, 1, self.chanNum, self.viewNum))).type(torch.FloatTensor) # batchSize*channel*512*360
        sino = sino.cuda()
        ''' rotate'''
        for i in range(self.viewNum):
            angle = - 180/self.viewNum*(i+1) * math.pi / 180 - math.pi
            A = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])  
            theta = np.array([[A[0, 0], A[0, 1], 0], [A[1, 0], A[1, 1], 0]])                                    
            theta = torch.from_numpy(theta).type(torch.FloatTensor)
            theta = theta.unsqueeze(0)
            theta = theta.repeat(self.batchSize,1,1)
            theta = theta.cuda()
            ''' interpolation'''
            grid = F.affine_grid(theta, x.size())
            x_rotate = F.grid_sample(x, grid) # 4*1*512*512
            ''' accumulation'''
            sino[:,:,:,i] = torch.sum(x_rotate, dim=2) 
        sino = sino*0.5
        sino = sino.cuda()    
        return sino


class FBP(nn.Module):
    def __init__(self, viewNum, chanNum, batchSize, netChanNum, chanSpacing):
        super(FBP, self).__init__()
        self.viewNum = viewNum # projection的投影角度数
        self.chanNum = chanNum # projection的通道数
        self.batchSize = batchSize
        self.netChanNum = netChanNum # 输入FBP网络数据的通道数
        self.chanSpacing = chanSpacing
    
    def forward(self, x):
        '''
            x:  projection (batchSize*netChanNum*chanNum*viewNum) 4*1*512*360
            type(x) is a tensor
        '''
        '''频域滤波'''
        projectionValue = convolution(x,self.batchSize,self.netChanNum,self.chanNum,self.viewNum,self.chanSpacing) # 2*1*512*360
        projectionValue = projectionValue.cuda()
        sino_rotate = np.zeros((self.batchSize, self.netChanNum, self.viewNum, self.chanNum, self.chanNum)) # batchSize*netChanNum*viewNum*chanNum*chanNum
        sino_rotate = torch.from_numpy(sino_rotate).type(torch.FloatTensor)
        sino_rotate = sino_rotate.cuda()
        AglPerView = math.pi/self.viewNum
        '''设置FOV,生成mask将FOV以外的区域置零'''
        FOV = torch.ones((self.batchSize,self.netChanNum,self.chanNum,self.chanNum))
        x_linespace = np.arange(1,self.chanNum+1,1)  # (512,)
        y_linespace = np.arange(1,self.chanNum+1,1)  # (512,)
        x_mesh,y_mesh = np.meshgrid(x_linespace,y_linespace) # 512*512
        XPos = (x_mesh-256.5) * self.chanSpacing # 512*512
        YPos = (y_mesh-256.5) * self.chanSpacing # 512*512
        R = np.sqrt(XPos**2 + YPos**2) # 512*512
        R = torch.from_numpy(R).type(torch.FloatTensor) # 512*512
        R = R.repeat(self.batchSize,self.netChanNum,1,1) # 2*1*512*512
        FOV[R>=self.chanSpacing*self.chanNum/2] = 0 # 2*1*512*512
        FOV = FOV.cuda()
        ''' rotate interpolation'''
        for i in range(self.viewNum):
            projectionValueFiltered = torch.unsqueeze(projectionValue[:,:,:,i],3) # 2*1*512*1
            projectionValueRepeat = projectionValueFiltered.repeat(1,1,1,512) # 2*1*512*512
            projectionValueRepeat = projectionValueRepeat * FOV  # 2*1*512*512
            angle = -math.pi/2 + 180/self.viewNum*(i+1) * math.pi / 180
            A = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
            theta = np.array([[A[0, 0], A[0, 1], 0], [A[1, 0], A[1, 1], 0]])
            theta = torch.from_numpy(theta).type(torch.FloatTensor)
            theta = theta.unsqueeze(0)
            theta = theta.cuda()
            theta = theta.repeat(self.batchSize,1,1)
            grid = F.affine_grid(theta, torch.Size((self.batchSize, self.netChanNum, 512, 512)))
            sino_rotate[:,:,i,:,:] = F.grid_sample(projectionValueRepeat, grid) # 512*512
        ''' accumulation'''
        iradon = torch.sum(sino_rotate, dim=2)  
        iradon = iradon*AglPerView
        return iradon


def convolution(proj,batchSize,netChann,channNum,viewnum,channSpacing):
    AglPerView = np.pi/viewnum
    channels = 512
    origin = np.zeros((batchSize,netChann,viewnum, channels, channels))
    # avoid truncation
    step = list(np.arange(0,1,1/100))
    step2 = step.copy()
    step2.reverse()
    step = np.array(step) # (100,)
    step = np.expand_dims(step,axis=1) # 100*1
    step = torch.from_numpy(step).type(torch.FloatTensor) # (100,1)
    step = step.repeat(batchSize,1,1,360) # 2*1*100*360
    step_temp = proj[:,:,0,:].unsqueeze(2) # 2*1*1*360
    step_temp = step_temp.repeat(1,1,100,1) # 2*1*100*360
    step = step.cuda()
    step = step*step_temp # 2*1*100*360
    step2 = np.array(step2) # (100,)
    step2 = np.expand_dims(step2,axis=1) # 100*1
    step2 = torch.from_numpy(step2).type(torch.FloatTensor) # (100,1)
    step2 = step2.repeat(batchSize,1,1,360) # 2*1*100*360
    step2_temp = proj[:,:,-1,:].unsqueeze(2) # 2*1*1*360
    step2_temp = step2_temp.repeat(1,1,100,1) # 2*1*100*360
    step2 = step2.cuda()
    step2 = step2*step2_temp # 2*1*100*360
    filterData = Ramp(batchSize,netChann,2*100+channNum,channSpacing) # 2*1*2048*360
    iLen = filterData.shape[2] # 2048
    proj = torch.cat((step,proj,step2),2) # 2*1*712*360
    proj = torch.cat((proj,torch.zeros(batchSize,netChann,iLen-proj.shape[2],viewnum).cuda()),2) # 2*1*2048*360
    sino_fft = fft(proj.detach().cpu().numpy(),axis=2) # 2*1*2048*360
    image_filter = filterData*sino_fft # 2*1*2048*360
    image_filter_ = ifft(image_filter,axis=2) # 2*1*2048*360
    image_filter_ = np.real(image_filter_)
    image_filter_ = torch.from_numpy(image_filter_).type(torch.FloatTensor)
    image_filter_final = image_filter_[:,:,100:512+100] # 2*1*512*360
    return image_filter_final

