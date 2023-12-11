/*
 * @Author: zjj
 * @Date: 2023-12-04 18:35:00
 * @LastEditors: zjj
 * @LastEditTime: 2023-12-04 18:35:08
 * @FilePath: /parking_perception_ros/src/multitask_det/architecture/avm_mlt_det/src/logger.cpp
 * @Description: 
 * 
 * Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
 */
#include "logger.h"
#include "logging.h"

Logger gLogger{Logger::Severity::kERROR};
LogStreamConsumer gLogVerbose{LOG_VERBOSE(gLogger)};
LogStreamConsumer gLogInfo{LOG_INFO(gLogger)};
LogStreamConsumer gLogWarning{LOG_WARN(gLogger)};
LogStreamConsumer gLogError{LOG_ERROR(gLogger)};
LogStreamConsumer gLogFatal{LOG_FATAL(gLogger)};

void setReportableSeverity(Logger::Severity severity)
{
    gLogger.setReportableSeverity(severity);
    gLogVerbose.setReportableSeverity(severity);
    gLogInfo.setReportableSeverity(severity);
    gLogWarning.setReportableSeverity(severity);
    gLogError.setReportableSeverity(severity);
    gLogFatal.setReportableSeverity(severity);
}