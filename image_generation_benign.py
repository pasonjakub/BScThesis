from bitstring import BitArray
from subprocess import call
import os
import shutil
from capstone import *
import pefile


def get_main_code_section(sections, base_of_code):      # function taken from https://isleem.medium.com/create-your-own-disassembler-in-python-pefile-capstone-754f863b2e1c
    addresses = []
    # get addresses of all sections
    for section in sections:
        addresses.append(section.VirtualAddress)

    # if the address of section corresponds to the first instruction then
    # this section should be the main code section
    if base_of_code in addresses:
        return sections[addresses.index(base_of_code)]
    # otherwise, sort addresses and look for the interval to which the base of code
    # belongs
    else:
        addresses.append(base_of_code)
        addresses.sort()
        if addresses.index(base_of_code) != 0:
            return sections[addresses.index(base_of_code) - 1]
        else:
            # this means we failed to locate it
            return None


def fine_disassemble(exe):      # function taken from https://isleem.medium.com/create-your-own-disassembler-in-python-pefile-capstone-754f863b2e1c
    # get main code section
    main_code = get_main_code_section(exe.sections, exe.OPTIONAL_HEADER.BaseOfCode)
    # define architecture of the machine
    output_data = ''
    md = Cs(CS_ARCH_X86, CS_MODE_32)
    md.detail = True
    last_address = 0
    last_size = 0
    # Beginning of code section
    begin = main_code.PointerToRawData
    # the end of the first continuous bloc of code
    end = begin + main_code.SizeOfRawData
    while True:
        # parse code section and disassemble it
        data = exe.get_memory_mapped_image()[begin:end]
        for i in md.disasm(data, begin):
            output_data += ''.join(format(x, '02x') for x in i.bytes)
            last_address = int(i.address)
            last_size = i.size
        # sometimes you need to skip some bytes
        begin = max(int(last_address), int(begin)) + last_size + 1
        if begin >= end:
            return output_data


extensions = ['.acm', '.ax', '.cpl', '.dll', '.drv', '.efi', '.exe', '.mui', '.ocx', '.scr', '.sys', '.tsp']
# extensions which are to be taken into consideration while generating images
directories = ['F://Intel', 'F://PerfLogs', 'F://Program Files', 'F://Program Files (x86)', 'F://ProgramData', 'F://Users', 'F://Windows']
# directories through which the program will walk in order to find all the files with above defined extensions
paths = []
destination_path = 'D://ImagesBenign'   # destination path where images are supposed to be saved
scriptPath = 'C://Users//Kuba//Desktop//bsc_thesis'  # directory where Bin2PNG.exe is run and the image are generated
counter = 0
errCounter = [0] * 3

for directory in directories:
    try:
        for subdir, dirs, files in os.walk(directory):
            for file in files:
                fileName, fileExtension = os.path.splitext(file)
                if fileExtension in extensions:
                    innerFile = os.path.join(subdir, file)
                    if os.path.getsize(innerFile) < 1000000:        # skipping files bigger than around 1MB in order to maintain speed with image generation
                        paths.append(innerFile)     # paths of files in specific directory
    except Exception as e:
        print(e)
    os.chdir(directory)
    for path in paths:  # looping through all the files
        counter += 1
        print(counter)  # counter to name files appropriately
        print(path)     # since this process is taking long time it is recommended to record specific place of progression in order to easily resume the process
        # if counter < x: continue # thanks to this condition we can skip any images which were already generated since the list 'paths' is ordered
        extension = path.split('.')[-1]
        hexFinal = ''
        try:
            # parse exe file
            pe = pefile.PE(path)
            try:
                # calling the function declared earlier
                hexFinal = fine_disassemble(pe)
                binFinal = BitArray(hex=hexFinal)
                try:
                    os.chdir(scriptPath)
                    with open('in.exe', 'wb') as f:
                        binFinal.tofile(f)
                    program = 'Bin2PNG.exe'
                    call([program, 'encrypt'])  # running the Bin2PNG.exe with 'encrypt' argument => (in.exe => image.png)
                    # Max/min size
                    if os.path.isfile(scriptPath + '//image.png'):  # if image was created correctly
                        shutil.move(scriptPath + '//image.png', destination_path + f'//{counter}_{extension}.png')
                except Exception as e:
                    print(e)
                    errCounter[0] += 1
                    pass  # skipping elements where there are permission problems
            except Exception as e:
                print(e)
                errCounter[1] += 1
                pass
        except Exception as e:
            print(e)
            errCounter[2] += 1
            pass
    print(directory)
print('0:', errCounter[0], '\n1:', errCounter[1], '\n2:', errCounter[2])        # print out how many exceptions were at specific place
