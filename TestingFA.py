import numpy as np
import mitsuba as mi
from IPython import embed
from time import time
from matplotlib import pyplot as plt
import FilterAlignment as FA

mi.set_variant('cuda_ad_rgb')

m2inch = 39.3701

def Beam_Prof_Test():
    filterxLoc = np.linspace(1,10,num = 5)/m2inch
    out = np.zeros((filterxLoc.shape[0],3))
    out[:,0] = filterxLoc


   

    sceneD = FA.Base_Scene(rad = 0.002,angle = 0.2,radiance = 1000000000)



    width = 500
    beamProf = np.zeros((out.shape[0],width))
    
    f,a = plt.subplots(1,1)

    for i in range(out.shape[0]):
        
        cfov = np.rad2deg(2*np.arctan2(0.1,filterxLoc[i])) #need to adjust fov so that pixel size is constant
        _,prof= FA.Render_Check(sceneD,[0.0,0,0],[filterxLoc[i],0,0],fov = cfov,illuminate = False,width = width)

        beamProf[i,:] = prof[:,0]
    
        a.plot(beamProf[i,:],label = 'Distance: %.2f m'%(filterxLoc[i]))

    a.set_xlabel('Pixel')
    a.set_ylabel('Irradiance[a.u.]')
    a.legend()

    plt.show()

    return 0 

def Beam_Drift_Test():
    maxFilt = 10
    filterxLoc = np.linspace(1,maxFilt,num = maxFilt)/m2inch
    out = np.zeros((filterxLoc.shape[0],3))
    out[:,0] = filterxLoc

    ang = 18


   

    sceneD = FA.Base_Scene()

    thickness = 0.001

    width = 500
    beamProf = np.zeros((out.shape[0],width))
    sign = np.ones_like(filterxLoc)
    #sign[1::2] = -1

    f,a = plt.subplots(1,1)


    for i in range(out.shape[0]):
        

        _,prof= FA.Render_Check(sceneD,[0.0,0,0],[filterxLoc[-1]+1/m2inch,0,0],fov = 20,illuminate = False,width = 500)
        

        beamProf[i,:] = prof[:,0]

    
        a.plot(beamProf[i,:],label = 'Num Filters: %d'%(i))
        FA.Add_Filter(sceneD,out[i,:],np.deg2rad(sign[i]*ang),name = 'Filter_'+str(i),thickness = thickness,type = 'filter')

    a.set_title('Filter Thickness: %.3f m'%(thickness))
    a.set_xlabel('Pixel')
    a.set_ylabel('Irradiance[a.u.]')
    a.legend()

    FA.Render_Check(sceneD,[0.0,0,0],[-0.1,0,0.1],fov = 50,illuminate = True)

    plt.show()

    return 0 

def Thickness_Test():
    filterxLoc = np.linspace(1,5,num = 5)/m2inch
    out = np.zeros((filterxLoc.shape[0],3))
    out[:,0] = filterxLoc

    

    width = 500
    beamProf = np.zeros((out.shape[0],width))
    sign = np.ones_like(filterxLoc)
    #sign[1::2] = -1
    f,a = plt.subplots(1,1)

    thickL = np.arange(0.001,0.005,step = 0.001)

    for i in range(thickL.shape[0]):

        
        sceneD = Base_Scene()

        for j in range(out.shape[0]):
            Add_Filter(sceneD,out[j,:],np.deg2rad(45),name = 'Filter_'+str(j),thickness = thickL[i],rough = False)
        

        _,prof= Render_Check(sceneD,[0.0,0,0],[filterxLoc[-1]+1/m2inch,0,0],fov = 20,illuminate = False,width = 500)
        

        beamProf[i,:] = prof[:,0]

    
        a.plot(beamProf[i,:],label = 'Filter Thickness: %.3f m'%(thickL[i]))


    a.set_title('Num Filters: %d'%(out.shape[0]))
    a.set_xlabel('Pixel')
    a.set_ylabel('Irradiance[a.u.]')
    a.legend()

    Render_Check(sceneD,[0.0,0,0],[-0.1,0,0.1],fov = 50,illuminate = True)

    plt.show()


    return 0 

def PMT_Test():
    filterxLoc = np.linspace(1,5,num = 5)/m2inch
    out = np.zeros((filterxLoc.shape[0],3))
    out[:,0] = filterxLoc


    #tar = Look_At(out, np.full(filterxLoc.shape,np.deg2rad(45)))

    sceneD = FA.Base_Scene(rad = 0.004,angle = 4,radiance = 100)

    """
    for i in range(out.shape[0]):
        FA.Add_Filter(sceneD,out[i,:],np.deg2rad(45),name = 'Filter_'+str(i),rough = True,mirror=True)
    """
    FA.Add_Filter(sceneD,out[0],np.deg2rad(15),name = 'Filter1',rough = True,mirror=False)
    FA.Add_PMT(sceneD,filterxLoc[0],-0.025,name  = 'PMT1',irrdMtr =True)

    
    #FA.Add_PMT(sceneD,filterxLoc[0],0.025,name  = 'PMT2')
    """
    sceneD['light'] = {
                'type': 'constant',
                'radiance': {
                    'type': 'rgb',
                    'value': 0.1,
                }
            }
    """
    t = time()
    print('Loading Scene')
    scene = mi.load_dict(sceneD)
    print('Scene Loaded in %f seconds'%(time()-t))   

    FA.Render_Check(sceneD,[filterxLoc[0],-0.025,0],[filterxLoc[0],0.1,0.1],\
                    fov = 50,illuminate = True)

    scene = FA.Modify_Filt_Pos(scene,'Filter1',np.deg2rad(60),[filterxLoc[0],0,0])

    FA.Render_Check(sceneD,[filterxLoc[0],-0.025,0],[filterxLoc[0],0.1,0.1],\
                    fov = 50,illuminate = True)

    plt.show()
    



    image = mi.render(scene,spp = 1000)

    
    print(np.asarray(image))
    
    f,a = plt.subplots(1,1)
    a.imshow(mi.util.convert_to_bitmap(image))
    #remove x and y ticks
    a.set_xticks([])
    a.set_yticks([])
    

    


    #FA.Render_Check(sceneD,[0.0,0,0],[-0.1,0,0.1],fov = 50,illuminate = False)
    FA.Render_Check(sceneD,[filterxLoc[0],-0.025,0],[filterxLoc[0],0.1,0.1],\
                    fov = 50,illuminate = False,spp = 500)
    FA.Render_Check(sceneD,[filterxLoc[0],-0.025,0],[filterxLoc[0],0.1,0.1],\
                    fov = 50,illuminate = True)

    plt.show()
    return 0 

def Modify_Test():

    filterxLoc = np.linspace(1,5,num = 5)/m2inch
    out = np.zeros((filterxLoc.shape[0],3))
    out[:,0] = filterxLoc


    #tar = Look_At(out, np.full(filterxLoc.shape,np.deg2rad(45)))

    sceneD = FA.Base_Scene(rad = 0.004,angle = 4,radiance = 100)

    target = [filterxLoc[0],-0.025,0]
    pupil = [filterxLoc[0],0.1,0.1]

    sceneD['camera'] = {
            'type': 'perspective',
            'fov': 50,
            'to_world': mi.ScalarTransform4f.look_at(origin=pupil,target=target,up=[0, 0, 1]),
            'film1': {
                'type': 'hdrfilm',
                'width':500, #low res for fast writing to numpy array
                'height': 500, 
                'pixel_format': 'luminance',
            },
    }

    sceneD['light'] = {
                'type': 'constant',
                'radiance': {
                    'type': 'rgb',
                    'value': 1.0,
                }
            }

    FA.Add_Filter(sceneD,out[0],np.deg2rad(15),name = 'Filter1',type = 'solid')

    t = time()
    print('Loading Scene')
    scene = mi.load_dict(sceneD)
    print('Scene Loaded in %f seconds'%(time()-t))

    image = mi.render(scene,spp = 100)

    
    f,a = plt.subplots(1,1)
    a.imshow(mi.util.convert_to_bitmap(image))
    #remove x and y ticks
    a.set_xticks([])
    a.set_yticks([])

    FA.Modify_Filt_Angle(scene,out[0],-30,'Filter1')

    image = mi.render(scene,spp = 100)


    
    f,a = plt.subplots(1,1)
    a.imshow(mi.util.convert_to_bitmap(image))
    #remove x and y ticks
    a.set_xticks([])
    a.set_yticks([])

    plt.show()

    return 0 

def Mirror_Scan():
    maxFilt = 14
    filterxLoc = np.linspace(1,maxFilt, num = maxFilt)/m2inch
    out = np.zeros((filterxLoc.shape[0],3))
    out[:,0] = filterxLoc

    ind = maxFilt-1


    #tar = Look_At(out, np.full(filterxLoc.shape,np.deg2rad(45)))

    sceneD = FA.Base_Scene(rad = 0.002,angle = 0.2,radiance = 1000000000)

    startAng = 45
    FA.Add_Filter(sceneD,out[ind],np.deg2rad(startAng),name = 'Filter1',type = 'mirror')
    FA.Add_PMT(sceneD,filterxLoc[ind],-0.025,name  = 'PMT1',irrdMtr =True)

    pupil = [filterxLoc[ind],0.1,0.1]
    target = [filterxLoc[ind],-0.025,0]

    pupil = [filterxLoc[ind],0.001,0.1]
    target = [filterxLoc[ind],0,0]
    sceneD['camera'] = {
            'type': 'perspective',
            'fov': 50,
            'to_world': mi.ScalarTransform4f().look_at(origin=pupil,target=target,up=[0, 0, 1]),
            'film1': {
                'type': 'hdrfilm',
                'width':100, #low res for fast writing to numpy array
                'height': 100, 
                'pixel_format': 'luminance',
            },
        }
    """
    sceneD['light'] = {
            'type': 'constant',
            'radiance': {
                'type': 'rgb',
                'value': 1.0,
            }
        }
    """
    

    

    t = time()
    print('Loading Scene')
    scene = mi.load_dict(sceneD)
    print('Scene Loaded in %f seconds'%(time()-t))

    #FA.Modify_Filt_eta(scene,'Filter1',0.9)



    #create a list of angles from -15 to 15
    angles = np.linspace(-5,5,num = 20)

    res= np.zeros(angles.shape[0])

    for i in range(angles.shape[0]):
        print(angles[i])
        scene = FA.Modify_Filt_Angle(scene,out[ind],angles[i],'Filter1',verbose = False)
        image = mi.render(scene,spp = 100000000,sensor = 0)
        res[i] = np.asarray(image)[0]
        """
        image = mi.render(scene,spp = 100, sensor = 1)
        f,a = plt.subplots(1,1)
        a.imshow(mi.util.convert_to_bitmap(image))
        #remove x and y ticks
        a.set_xticks([])
        a.set_yticks([])
        """
        
        
        

        scene = FA.Modify_Filt_Angle(scene,out[ind],-1*angles[i],'Filter1') #since it is a relative angle need to reset it


    f,a  = plt.subplots(1,1)
    a.plot(angles+45,res)
    a.set_xlabel('Angle[deg]')
    a.set_ylabel('Irradiance[a.u.]')
    plt.show()
    return 0 


def Baseline_Test():
    maxFilt = 14
    filterxLoc = np.linspace(1,maxFilt,num = maxFilt)/m2inch
    out = np.zeros((filterxLoc.shape[0],3))
    out[:,0] = filterxLoc


    #tar = Look_At(out, np.full(filterxLoc.shape,np.deg2rad(45)))
    sceneD = FA.Base_Scene(rad = 0.002,angle = 0.2,radiance = 1000000000)

    fNameL = []
    senNameL = []
    for i in range(out.shape[0]):
        fName = 'Filter_'+str(i)
        sName = 'PMT_'+str(i)
        FA.Add_Filter(sceneD,out[i,:],np.deg2rad(45),name = fName, type = 'mirror')
        FA.Add_PMT(sceneD,filterxLoc[i],-0.025,name  = sName,irrdMtr =True)

        fNameL.append(fName)
        senNameL.append(sName)
    """
    FA.Render_Check(sceneD,[filterxLoc[2],-0.025,0],[filterxLoc[2],0.1,0.1],\
                fov = 60,illuminate = True)
    
    FA.Render_Check(sceneD,[filterxLoc[5],0,0],[0,0.0001,0.1],\
                fov = 60,illuminate = True)
    
    plt.show()
    """

    FA.Baseline_Report(sceneD,fNameL,senNameL,plot = True)
    

def T_Align_Pert():

    filterxLoc = np.linspace(1,5,num = 1)/m2inch
    out = np.zeros((filterxLoc.shape[0],3))
    out[:,0] = filterxLoc


    #tar = Look_At(out, np.full(filterxLoc.shape,np.deg2rad(45)))
    sceneD = FA.Base_Scene(rad = 0.002,angle = 3,radiance = 100)

    fNameL = []
    senNameL = []
    for i in range(out.shape[0]-1):
        fName = 'Filter_'+str(i)
        sName = 'PMT_'+str(i)
        FA.Add_Filter(sceneD,out[i,:],np.deg2rad(45),name = fName, type = 'filter')
        FA.Add_PMT(sceneD,filterxLoc[i],-0.025,name  = sName,irrdMtr =True)

        fNameL.append(fName)
        senNameL.append(sName)

    #now add in the final PMT and mirror which we will measure
    iMirr = out.shape[0]-1
    fName = 'Filter_'+str(iMirr)
    sName = 'PMT_'+str(iMirr)
    FA.Add_Filter(sceneD,out[iMirr,:],np.deg2rad(45),name = fName, type = 'mirror')
    FA.Add_PMT(sceneD,filterxLoc[iMirr],-0.025,name  = sName,irrdMtr =True)

    fNameL.append(fName)
    senNameL.append(sName)

    scene = mi.load_dict(sceneD)


    FA.Align_Pert(scene,fNameL,out,iMirr,0.5)

    return 0 

def Beam_Expansion_Test():

    sceneD = FA.Base_Scene(rad = 0.002,angle = 0.1,radiance = 100000000)
    """
    FA.Add_Filter(sceneD,[0.05,0,0],np.deg2rad(90), type = 'rough')

    _,row = FA.Render_Check(sceneD,[0,0,0],[0.1,0,0],\
                fov = 60,illuminate = False)
    """

    FA.Add_Filter(sceneD,[0.1,0,0],np.deg2rad(90), type = 'rough')

    _,row = FA.Render_Check(sceneD,[0.1,0,0],[0,0,0],\
                fov = 20,illuminate = False)
    
    f,a = plt.subplots(1,1)
    a.plot(row)
    plt.show()
    
    embed()
    
    plt.show()

    return 0 

def Unangled_Test():
    sceneD = FA.Base_Scene(rad = 0.002,angle = 0.2,radiance = 1000000000)

    maxFilt =10
    filterxLoc = np.linspace(1,maxFilt,num = maxFilt)/m2inch
    #filterxLoc = np.linspace(1,maxFilt,num = maxFilt)
    out = np.zeros((filterxLoc.shape[0],3))
    out[:,0] = filterxLoc

    sign = np.ones_like(filterxLoc)
    sign[1::2] = -1 #comment out to make all one direction
    print(sign)


    fNameL = []
    senNameL = []
    for i in range(out.shape[0]-1):
        fName = 'Filter_'+str(i)
        sName = 'PMT_'+str(i)
        FA.Add_Filter(sceneD,out[i,:],np.deg2rad(sign[i]*45),name = fName, type = 'filter',thickness = 0.005)
        #FA.Add_PMT(sceneD,filterxLoc[i],-0.025,name  = sName,irrdMtr =True)

        fNameL.append(fName)

    iMirr = out.shape[0]-1
    fName = 'Filter_'+str(iMirr)
    sName = 'PMT_'+str(iMirr)
    FA.Add_Filter(sceneD,out[iMirr,:],np.deg2rad(sign[iMirr]*45),name = fName, type = 'mirror')
    FA.Add_PMT(sceneD,filterxLoc[iMirr],sign[iMirr]*-0.025,name  = sName,irrdMtr =True)
    """
    _,row = FA.Render_Check(sceneD,[filterxLoc[-1],0,0],[0,0,0.1],\
            fov = 45,illuminate = True)
    plt.show()
    """
    

    scene = mi.load_dict(sceneD)



    surf = scene.sensors()[0].get_shape().surface_area()

    image = mi.render(scene,spp = 100000000)

    print('resulting signal')
    print(np.asarray(image)[0]*surf)


    angles = np.linspace(-5,4,num = 20)


    res= np.zeros(angles.shape[0])

    for i in range(angles.shape[0]):
        scene = FA.Modify_Filt_Angle(scene,out[iMirr],angles[i],fName,verbose = False)
        image = mi.render(scene,spp = 100000000)
        res[i] = np.asarray(image)[0]*surf


        scene = FA.Modify_Filt_Angle(scene,out[iMirr],-1*angles[i],fName) #since it is a relative angle need to reset it

    f,a  = plt.subplots(1,1)
    a.plot(angles+45,res)
    a.set_xlabel('Angle[deg]')
    a.set_ylabel('Irradiance[a.u.]')
    plt.show()



    
    plt.show()
    return 0 

def Low_Angle_Check():

    sceneD = FA.Base_Scene(rad = 0.002,angle = 0.2,radiance = 1000000000)

    maxFilt =10
    filterxLoc = np.linspace(1,maxFilt,num = maxFilt)*0.04
    #filterxLoc = np.linspace(1,maxFilt,num = maxFilt)
    out = np.zeros((filterxLoc.shape[0],3))
    out[:,0] = filterxLoc

    sign = 1
    ang = 18 #deg
    distFPMT = 0.05 #m


    y = distFPMT*np.sin(np.deg2rad(ang*2)) #distance off axis
    x = distFPMT*np.cos(np.deg2rad(ang*2)) #distancef from Filter

    x = filterxLoc-np.full(out.shape[0],x)

    PMTPos = np.vstack((x,-1*np.full(out.shape[0],y),np.zeros_like(filterxLoc))).T

    """
    fOut = out[:-1]
    FA.Filter_Chain(sceneD,fOut,np.full(fOut.shape[0],ang),type = 'filter',\
                    alternate = False,ptPMT = PMTPos[:-1],PMT = False)
    """

    iMirr = out.shape[0]-1
    iMirr =0
    fName = 'Filter_'+str(iMirr)
    sName = 'PMT_'+str(iMirr)
    FA.Add_Filter(sceneD,out[iMirr,:],np.deg2rad(sign*18),name = fName, type = 'mirror')
    FA.Add_PMT(sceneD,PMTPos[iMirr,0],sign*PMTPos[iMirr,1],name  = sName,irrdMtr =True,xTar = filterxLoc[iMirr])

    """
    _,row = FA.Render_Check(sceneD,[filterxLoc[maxFilt//2],0,0],[-0.1,0,0.2],width = 100,\
            fov = 70,illuminate = True)
    
    _,row = FA.Render_Check(sceneD,[filterxLoc[maxFilt//2],0,0],[filterxLoc[maxFilt//2],0.001,0.3],width = 500,\
        fov = 70,illuminate = True)
    plt.show()
    """

    scene = mi.load_dict(sceneD)



    surf = scene.sensors()[0].get_shape().surface_area()

    image = mi.render(scene,spp = 100000000)

    print('resulting signal')
    print(np.asarray(image)[0]*surf)


    angles = np.linspace(-3,3,num = 20)


    res= np.zeros(angles.shape[0])

    for i in range(angles.shape[0]):
        scene = FA.Modify_Filt_Angle(scene,out[iMirr],angles[i],fName,verbose = False)
        image = mi.render(scene,spp = 100000000)
        res[i] = np.asarray(image)[0]*surf


        scene = FA.Modify_Filt_Angle(scene,out[iMirr],-1*angles[i],fName) #since it is a relative angle need to reset it

    f,a  = plt.subplots(1,1)
    a.plot(angles+ang,res)
    a.set_xlabel('Angle[deg]')
    a.set_ylabel('Irradiance[a.u.]')
    plt.show()


    

    return 0 


if __name__ == '__main__':
    #PMT_Test()
    #Modify_Test()
    #Mirror_Scan()
    #Baseline_Test()
    #T_Align_Pert()

    #Beam_Expansion_Test()

    #Unangled_Test()

    #Beam_Drift_Test()
    Low_Angle_Check()