import numpy as np
import mitsuba as mi
import drjit as dr
from IPython import embed
from time import time
from matplotlib import pyplot as plt

from scipy.stats import binned_statistic

mi.set_variant('cuda_ad_rgb')

m2inch = 39.3701

def Look_At(pt, angle):
    #assumes the angle is that between the normal of the filter and the x axis
    pt = np.atleast_2d(pt)
    angle = np.atleast_1d(angle)
    xpt = pt[:,0]
    ypt = pt[:,1]

    outPt = xpt*np.tan(angle) 

    out = np.zeros((angle.shape[0],3))
    out[:,1] = ypt+outPt



    return out

def Generate_Angle(SD, center=0, num = 50):
    #randomly generate an angle from a normal distribution centered at center
    # with a standard deviation of SD
    #Inputs:
    #SD: float, the standard deviation of the normal distribution in degrees
    #center: float, the center of the normal distribution in degrees
    #num: int, the number of angles to generate

    return np.random.normal(center,SD,num)

def Add_Filter(sceneD,pt,angle, thickness = 0.001,name = None,int_ior = 1.5,type = 'rough'):
    #assumes scene units is in meters

    if name is None:
        name = "Filter"

    if type=='rough':


        sceneD[name] = {'type': 'cube',
                        'to_world': mi.ScalarTransform4f.look_at(origin=pt,target=Look_At(pt,angle)[0],up=[0, 0, 1])@mi.ScalarTransform4f.scale([thickness/2,1/(2*m2inch),1/(2*m2inch)]),
                        'bsdf': {'type': 'roughdielectric',
                                'int_ior': 1.5,
                                'distribution': 'beckmann',
                                'alpha': 0.8,
                                },
                        

        }
    elif type =='mirror':

        sceneD[name] = {'type': 'cube',
                        'to_world': mi.ScalarTransform4f.look_at(origin=pt,target=Look_At(pt,angle)[0],up=[0, 0, 1])@mi.ScalarTransform4f.scale([thickness/2,1/(2*m2inch),1/(2*m2inch)]),
                        'bsdf': {'type': 'conductor',
                                'material': 'none',
                                },
                        

        }
    elif type =='solid':

        sceneD[name] = {'type': 'cube',
                        'to_world': mi.ScalarTransform4f.look_at(origin=pt,target=Look_At(pt,angle)[0],up=[0, 0, 1])@mi.ScalarTransform4f.scale([thickness/2,1/(2*m2inch),1/(2*m2inch)]),
                        'bsdf': {'type': 'roughconductor',
                                'material': 'W',
                                'alpha': 0.8,
                                },
                        

        }
    else:
        sceneD[name] = {'type': 'cube',
                    'to_world': mi.ScalarTransform4f.look_at(origin=pt,target=Look_At(pt,angle)[0],up=[0, 0, 1])@mi.ScalarTransform4f.scale([thickness/2,1/(2*m2inch),1/(2*m2inch)]),
                    'bsdf': {'type': 'dielectric',
                            'int_ior': 1.5,
                            },
                    

                    }

    return 0


def Modify_Filt_Angle(scene,pt,angle,name):
    #modifies the position of a filter, specified by name, in the mitsuba scene change the angle by angle degrees
    #Inputs:
    #scene: mitsuba scene object, the scene object
    #pt: len(3) array, the center of the filter
    #angle: float, the angle to rotate the filter by in degrees, this is a relative shift to the initial angle
    #name: string, the name of the filter object in the scene
    #Returns:
    #scene: mitsuba scene object, the modified scene object


    params = mi.traverse(scene)

    vpos = params[name+'.vertex_positions']
    vnorm = params[name+'.vertex_normals']

    tL = [vpos,vnorm]


    #trans = mi.Transform4f.look_at(origin=pt,target=Look_At(pt,angle)[0],up=[0, 0, 1])
    #first move to the origin
    translate = mi.Transform4f.translate(-pt)
    rotate = mi.Transform4f.rotate([0,0,1],angle = angle)
    #return to the original position
    transback = mi.Transform4f.translate(pt)
    trans = transback@rotate@translate

    for i,item in enumerate(tL):
        
        vposUR = dr.unravel(dr.cuda.ad.Array3f,item, order = 'F')
        vposT = trans@vposUR
        tL[i] = dr.ravel(vposT, order = 'F')

    params[name+'.vertex_positions'] = tL[0]
    params[name+'.vertex_normals'] = tL[1]

    params.update()
    return scene

def Modify_Filt_eta(scene,name, eta):
    #modifies the index of refraction of a filter in the mitsuba scene
    #only works for dielectric filters
    
    params = mi.traverse(scene)
    params[name+'.bsdf.eta'] = eta
    params.update()
    return scene

def Add_Column_Light(sceneD,radius,angle,radiance=100.0):
    #places a area light inside a cylinder so that it is collimated
    # the length of the cylinder is set by the desired angle of the output light
    #the output of the cylinder is placed at the origin and the light is placed at the center of the cylinder
    # a distance length along the z-axis
    #Inputs:
    #sceneD: dictionary, the scene dictionary
    #radius: float, the radius of the cylinder in meters or consistent with the scene
    #angle: float, the half-angle of the collimated light in degrees
    #radiance : float, the radiance of the light source, Power/Area/sr
    #returns:
    #sceneD: dictionary, the modified scene dictionary

    angle = np.deg2rad(angle)
    length = radius/np.tan(angle)

    delta = length/1000.0


    sceneD['light'] = { 'type': 'disk',
                            'to_world': mi.ScalarTransform4f.look_at(origin=[-length+delta,0,0],target=[1,0,0],up=[0, 0, 1])@mi.ScalarTransform4f.scale(radius-delta),
                            'emitter': {'type': 'area',
                                       'radiance':{'type': 'spectrum',
                                                   'value' : radiance,
                                                   },
                                       },
                            }
    #cylinder is placed along the z-axis
    # it is perfectly absorbing on the inside by default
    sceneD['light_cylinder'] = {'type': 'cylinder',
                                'p0': [-length,0,0],
                                'p1': [0,0,0],
                                'radius': radius,
                                'flip_normals':True,
                                'bsdf': {'type': 'diffuse',
                                        'reflectance':0.0,
                                        },
                                }
    
    #now add in an end cap to the cylinder
    sceneD['light_endcap'] = {'type': 'disk',
                            'to_world': mi.ScalarTransform4f.look_at(origin=[-length,0,0],target=[0,0,0],up=[0, 0, 1])@mi.ScalarTransform4f.scale(radius),
                            'bsdf': {'type': 'diffuse',
                                    'reflectance':0.0,
                                    },
                            }
    
    return sceneD


def Base_Scene(rad = 0.002,angle = 4, radiance = 1000):

    base_scene = {'type': 'scene',
                  'integrator': {'type': 'volpath',
                                 'max_depth':-1,
                                 'hide_emitters': False,
                                 },
                
    }

    base_scene = Add_Column_Light(base_scene,rad,angle,radiance)

    return base_scene 

def Add_PMT(sceneD,xCent,yCent,name = None,width = 500,irrdMtr = False):

    #adds a CAD model representaiton of a PMT, just a rectangular box wit a hole in one face
    #as well as a camera sensor inside the box facing the hole
    #assumes the PMT is centered in the z plane 
    #Inputs:
    #sceneD: dictionary, the scene dictionary
    #xCent: float, the x center of hole of the PMT in meters
    #yCent: float, the y center of the hole of the PMT in meters
    #name: string, the name of the PMT object in the scene dictionary
    #width: int, the width of the camera sensor in pixels
    #Returns:
    #sceneD: dictionary, the modified scene dictionary

    if name is None:
        name = 'PMT'

    PMTfaceDim = 0.022 #m
    PMTlongDim = 0.05 #m

    yDir = 1
    if yCent < 0:
        yDir = -1

    cx = xCent
    cy = yCent
    

    sceneD[name] = {'type': 'obj',
                     'filename' : 'PMTmitsuba.obj',
                     'to_world': mi.ScalarTransform4f.look_at(origin=[cx,cy,0],target=[cx,0,0],up=[0, 0, 1])@mi.ScalarTransform4f.scale(0.001), #convert from mm to m
                     'face_normals': True,
                     'bsdf': {'type': 'roughconductor',
                                'material': 'W',
                                'alpha': 0.8,
                                },
    }
    
    

    camy = yCent+yDir*PMTfaceDim/2.0

    if irrdMtr:
        sceneD[name+'_sensor'] = {'type': 'disk',
                                  'to_world': mi.ScalarTransform4f.look_at(origin=[cx,camy+yDir*0.0005,0],target=[cx,0,0],up=[0, 0, 1])@mi.ScalarTransform4f.scale(0.0045),

                                  'sensor': {'type': 'irradiancemeter',
                                        'film':{
                                        'type': 'hdrfilm',
                                        'pixel_format': 'luminance',
                                        'filter':{'type':'box'},
                                        'width':1,
                                        'height':1,
                                        },
                                    },
        }

    else:
    
        #no offset in z for camera since it is defined by its center point
        sceneD[name+'_sensor'] = {'type': 'perspective',
                                'fov': 90,
                                'to_world': mi.ScalarTransform4f.look_at(origin=[cx,camy,0],target=[cx,0,0],up=[0, 0, 1]), 
                                'sensor': {
                                    'type': 'hdrfilm',
                                    'width':width, #low res for fast writing to numpy array
                                    'height': width, 
                                    'pixel_format': 'luminance',
                                },
        }
    

    return sceneD

def AddSensor(sceneD, name, loc, tar):
    sceneD[name] = {'type': 'disk',
                'to_world': mi.ScalarTransform4f.look_at(origin=loc,target=tar,up=[0, 0, 1])@mi.ScalarTransform4f.scale(0.004), #4mm radius sensor
                'sensor': {
                    'type': 'irradiancemeter',
                        'film':{
                        'type': 'hdrfilm',
                        'pixel_format': 'luminance',
                        'filter':{'type':'box'},
                        'width':1,
                        'height':1,
                        } 
                    }
                }
    
    return sceneD

def Align_Pert(scene, filterNL, filterPts, sensorI, SD,num=50,plot = True):
    #perturbs the filters specified by filterNL and filterPts in the scene dictionary by a random angle
    #choosen from a normal distribution with a standard deviation of SD
    #renders the scene with the perturbed filters and returns the irradiance at the sensor specified by sensorI
    #Inputs:
    #scene: mitsuba scene object, the scene object
    #filterNL: list of strings, the names of the filters to perturb
    #filterPts: list of len(3) arrays, the center points of the filters
    #sensorI: int, the index of the sensor in the scene object to measure the irradiance at
    #SD: float, the standard deviation of the normal distribution of the angle perturbation in degrees
    #num: int, the number of simulations to run
    #plot: bool, if true plots a histogram of the perturbations
    #Returns:
    #out: len(num) array, the irradiance at the sensor for each simulation

    totNum = len(filterNL)*num

    angles = Generate_Angle(SD, center=0, num = totNum)

    angles = angles.reshape((len(filterNL),num))

    out = np.zeros(num)

    baseline = np.asarray(mi.render(scene,spp = 100000,sensor = sensorI))[0]


    for i in range(num):
        for j,cFilter in enumerate(filterNL):
            scene = Modify_Filt_Angle(scene,filterPts[j],angles[j,i],cFilter)

        image = mi.render(scene,spp = 100000,sensor = sensorI)
        out[i]= np.asarray(image)[0]
        print(out[i])

        #now reset the angles
        for j,cFilter in enumerate(filterNL):
            scene = Modify_Filt_Angle(scene,filterPts[j],-1*angles[j,i],cFilter)

    if plot:
        f,a = plt.subplots(1,1)
        aveDev = np.mean(angles,axis = 0)
        nbins = 10
        dat, bin_edge,_ = binned_statistic(aveDev,out,bins = nbins,statistic = 'mean',range = (-3*SD,3*SD))
        minD, _,_ = binned_statistic(aveDev,out,bins = nbins,statistic = 'min',range = (-3*SD,3*SD))
        maxD,_,_ = binned_statistic(aveDev,out,bins = nbins,statistic = 'max',range = (-3*SD,3*SD))

        xaxis = (bin_edge[1:] + bin_edge[:-1]) / 2 #get the middle of each bin
        a.errorbar(xaxis,dat,yerr = [dat-minD,maxD-dat],fmt = 'o')


        a.set_xlabel('Average Angle Deviation [deg]')
        a.set_ylabel('Average Irradiance[a.u.]')

        #horizontal line at the max irradiance
        a.axhline(baseline,color = 'r',label = 'Max Irradiance')

        plt.show()

    return 0 



def Baseline_Report(sceneD, filterNL,senNameL,plot = False):
    #renders the scene with each filter as a mirror to get a baseline irradiance 
    #for comparison in the filter alignment test

    #copy the scene dictionary to not modify the original
    sceneCopy = sceneD.copy()


    out = np.zeros(len(filterNL))

    for i,cFilter in enumerate(filterNL):
        #assumes that the filters are ordered by closeness to the light source

        scene = mi.load_dict(sceneCopy)
        image = mi.render(scene,spp = 1000000,sensor = i) #hopefully sensors are indexed by order added to scene
        #scene.sensors()[i].shape().surface_area()
        out[i] = np.asarray(image)[0]

        sceneCopy.pop(cFilter)
        #sceneCopy.pop(senNameL[i])
        #sceneCopy.pop(senNameL[i]+'_sensor')
        

    if plot:
        f,a = plt.subplots(1,1)
        a.plot(out)
        a.set_xlabel('Filter Number')
        a.set_ylabel('Irradiance[a.u.]')
        plt.show()

    return out





    return 0 

def Render_Check(sceneD,target, pupil=[0,0,0],width=500,fov=27,illuminate = True,spp = 100):
    """
    Renders the scene and displays the image by creating a camera and a diffuse light source.
    Does not modify the scene dictionary. Must call plt.show() to display the image after calling this funciton
    Inputs:
    sceneD: dictionary, the scene dictionary
    target: len(3) aray, the target for the camera
    pupil: len(3) array, the position of the camera
    width: int, the width of the image in pixels, foced square
    fov: float, the field of view of the camera
    illuminate: bool, if true add a diffuse light source to the scene, if False a light source should already be present in sceneD
    Returns:
    tDict: dictionary, a copy of the scene dictionary with the camera and light source added
    """

    # Irradiance objects must be removed from the scene
    # before rendering (alternatively if using mi scene object can specify the film to be redenered possibly)

    tDict = sceneD.copy() #copy the dicitonary to not modify it

    #remove the sensor objects
    sensors = []
    for k in tDict.keys():
        if type(tDict[k]) is dict:
            if 'sensor' in tDict[k].keys():
                sensors.append(k)
    for k in sensors:
        tDict.pop(k)
                


    
    tDict['camera'] = {
            'type': 'perspective',
            'fov': fov,
            'to_world': mi.ScalarTransform4f.look_at(origin=pupil,target=target,up=[0, 0, 1]),
            'film1': {
                'type': 'hdrfilm',
                'width':width, #low res for fast writing to numpy array
                'height': width, 
                'pixel_format': 'luminance',
            },
        }
    if illuminate:

        #add in a diffuse light source
        tDict['light'] = {
                'type': 'constant',
                'radiance': {
                    'type': 'rgb',
                    'value': 1.0,
                }
            }
        

    

    t = time()
    print('Loading Scene')
    scene = mi.load_dict(tDict)
    print('Scene Loaded in %f seconds'%(time()-t))

    image = mi.render(scene,spp = spp)
    print(image)

    f,a = plt.subplots(1,1)
    a.imshow(mi.util.convert_to_bitmap(image))
    #remove x and y ticks
    a.set_xticks([])
    a.set_yticks([])

    imAr = np.asarray(image)

    #take a row at the center of the image and plot it
    row = imAr[imAr.shape[0]//2,:]


    return tDict, row


if __name__ == '__main__':
    #Beam_Drift_Test()
    #Thickness_Test()
    PMT_Test()

    math.sqrt(-1)

    ### TESTING ###
    filterxLoc = np.linspace(1,5,num = 5)/m2inch
    out = np.zeros((filterxLoc.shape[0],3))
    out[:,0] = filterxLoc


    #tar = Look_At(out, np.full(filterxLoc.shape,np.deg2rad(45)))

    sceneD = Base_Scene()
    """
    for i in range(out.shape[0]):
        Add_Filter(sceneD,out[i,:],np.deg2rad(45),name = 'Filter_'+str(i))
    """


    #AddSensor(sceneD,'sensor',[6/m2inch,0,0],[0,0,0])

    #Render_Check(sceneD,[0.0,0,0],[-0.1,0,0.1],fov = 50,illuminate = True)
    width = 500
    beamProf = np.zeros((out.shape[0],width))
    f,a = plt.subplots(1,1)

    for i in range(out.shape[0]):
        #Add_Filter(sceneD,out[i,:],np.deg2rad(45),name = 'Filter_'+str(i))
        cfov = np.rad2deg(2*np.arctan2(0.1,filterxLoc[i])) #need to adjust fov so that pixel size is constant
        _,prof= Render_Check(sceneD,[0.0,0,0],[filterxLoc[i],0,0],fov = cfov,illuminate = False,width = width)

        beamProf[i,:] = prof[:,0]

    
        a.plot(beamProf[i,:],label = 'Distance: %.2f m'%(filterxLoc[i]))

    a.set_xlabel('Pixel')
    a.set_ylabel('Irradiance[a.u.]')
    a.legend()
    plt.show()

    """
    t = time()
    print('Loading Scene')
    scene = mi.load_dict(sceneD)
    print('Scene Loaded in %f seconds'%(time()-t))

    im = mi.render(scene,spp = 10000)
    """




